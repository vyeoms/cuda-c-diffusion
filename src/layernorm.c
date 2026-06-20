#include "nn.h"
#include <stdlib.h>

typedef struct {
    int C, max_batch;
    float eps;
    float *gamma, *beta; /* affine params */
    float *ggamma, *gbeta; /* their grads */
    float *mu, *rstd; /* per-row stats cached at forward */
} LayerNormState;

static void layernorm_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    LayerNormState* st = (LayerNormState*)self->state;
    launch_layernorm_forward(out, st->mu, st->rstd, in, st->gamma, st->beta,
        batch, st->C, st->eps, ctx->stream);
}

static void layernorm_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)out;
    LayerNormState* st = (LayerNormState*)self->state;
    launch_layernorm_backward(grad_in, st->ggamma, st->gbeta, grad_out, in,
        st->gamma, st->mu, st->rstd, batch, st->C, ctx->stream);
}

static void layernorm_parameters(Module* self, ParamList* pl)
{
    LayerNormState* st = (LayerNormState*)self->state;
    param_list_add(pl, st->gamma, st->ggamma, st->C);
    param_list_add(pl, st->beta, st->gbeta, st->C);
}

static void layernorm_free(Module* self)
{
    LayerNormState* st = (LayerNormState*)self->state;
    cudaFree(st->gamma);
    cudaFree(st->beta);
    cudaFree(st->ggamma);
    cudaFree(st->gbeta);
    cudaFree(st->mu);
    cudaFree(st->rstd);
    free(st);
    free(self);
}

Module* layernorm_create(int dim, int max_batch)
{
    Module* m = (Module*)malloc(sizeof(Module));
    LayerNormState* st = (LayerNormState*)malloc(sizeof(LayerNormState));
    st->C = dim;
    st->max_batch = max_batch;
    st->eps = 1e-5f;
    st->gamma = nn_device_alloc(dim);
    st->beta = nn_device_alloc(dim);
    st->ggamma = nn_device_alloc(dim);
    st->gbeta = nn_device_alloc(dim);
    st->mu = nn_device_alloc(max_batch);
    st->rstd = nn_device_alloc(max_batch);

    float* ones = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; ++i)
        ones[i] = 1.0f;
    NN_CUDA_CHECK(cudaMemcpy(st->gamma, ones, dim * sizeof(float), cudaMemcpyHostToDevice));
    NN_CUDA_CHECK(cudaMemset(st->beta, 0, dim * sizeof(float)));
    free(ones);

    m->name = "layernorm";
    m->in_dim = dim;
    m->out_dim = dim;
    m->state = st;
    m->forward = layernorm_forward;
    m->backward = layernorm_backward;
    m->parameters = layernorm_parameters;
    m->free = layernorm_free;
    return m;
}
