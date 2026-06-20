#include "nn.h"
#include <stdlib.h>

typedef struct {
    int T, C; /* num_embeddings, dim */
    float *table, *gtable;
} EmbedState;

static void embed_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    EmbedState* st = (EmbedState*)self->state;
    launch_embed_gather(out, st->table, in, batch, st->C, st->T, ctx->stream);
}

static void embed_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)out;
    (void)grad_in; /* index input carries no gradient */
    EmbedState* st = (EmbedState*)self->state;
    /* grad accumulates into gathered rows, so zero it first */
    NN_CUDA_CHECK(cudaMemsetAsync(st->gtable, 0,
        (size_t)st->T * st->C * sizeof(float), ctx->stream));
    launch_embed_scatter(st->gtable, grad_out, in, batch, st->C, st->T, ctx->stream);
}

static void embed_parameters(Module* self, ParamList* pl)
{
    EmbedState* st = (EmbedState*)self->state;
    param_list_add(pl, st->table, st->gtable, st->T * st->C);
}

static void embed_free(Module* self)
{
    EmbedState* st = (EmbedState*)self->state;
    cudaFree(st->table);
    cudaFree(st->gtable);
    free(st);
    free(self);
}

Module* learned_embed_create(Context* ctx, int num_embeddings, int dim)
{
    Module* m = (Module*)malloc(sizeof(Module));
    EmbedState* st = (EmbedState*)malloc(sizeof(EmbedState));
    st->T = num_embeddings;
    st->C = dim;
    st->table = nn_device_alloc((size_t)num_embeddings * dim);
    st->gtable = nn_device_alloc((size_t)num_embeddings * dim);
    nn_fill_normal(ctx->curand, st->table, (size_t)num_embeddings * dim, 0.0f, 1.0f);

    m->name = "learned_embed";
    m->in_dim = 1;
    m->out_dim = dim;
    m->state = st;
    m->forward = embed_forward;
    m->backward = embed_backward;
    m->parameters = embed_parameters;
    m->free = embed_free;
    return m;
}
