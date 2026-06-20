#include "nn.h"
#include <stdlib.h>

typedef struct {
    int C, G, H, W, max_batch;
    float eps;
    float *gamma, *beta;
    float *ggamma, *gbeta;
    float *mu, *rstd;
} GroupNormState;

static void groupnorm_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    GroupNormState* st = (GroupNormState*)self->state;
    launch_groupnorm_forward(out, st->mu, st->rstd, in, st->gamma, st->beta,
        batch, st->H * st->W, st->C, st->G,
        st->eps, ctx->stream);
}

static void groupnorm_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)out;
    GroupNormState* st = (GroupNormState*)self->state;
    launch_groupnorm_backward(grad_in, st->ggamma, st->gbeta, grad_out, in,
        st->gamma, st->mu, st->rstd,
        batch, st->H * st->W, st->C, st->G, ctx->stream);
}

static void groupnorm_parameters(Module* self, ParamList* pl)
{
    GroupNormState* st = (GroupNormState*)self->state;
    param_list_add(pl, st->gamma, st->ggamma, st->C);
    param_list_add(pl, st->beta, st->gbeta, st->C);
}

static void groupnorm_free(Module* self)
{
    GroupNormState* st = (GroupNormState*)self->state;
    cudaFree(st->gamma);
    cudaFree(st->beta);
    cudaFree(st->ggamma);
    cudaFree(st->gbeta);
    cudaFree(st->mu);
    cudaFree(st->rstd);
    free(st);
    free(self);
}

Module* spatial_groupnorm_create(int C, int num_groups, int H, int W, int max_batch)
{
    Module* m = (Module*)malloc(sizeof(Module));
    GroupNormState* st = (GroupNormState*)malloc(sizeof(GroupNormState));
    st->C = C;
    st->G = num_groups;
    st->H = H;
    st->W = W;
    st->max_batch = max_batch;
    st->eps = 1e-5f;

    st->gamma = nn_device_alloc(C);
    st->beta = nn_device_alloc(C);
    st->ggamma = nn_device_alloc(C);
    st->gbeta = nn_device_alloc(C);
    st->mu = nn_device_alloc((size_t)max_batch * num_groups);
    st->rstd = nn_device_alloc((size_t)max_batch * num_groups);

    float* ones = (float*)malloc(C * sizeof(float));
    for (int i = 0; i < C; ++i)
        ones[i] = 1.0f;
    NN_CUDA_CHECK(cudaMemcpy(st->gamma, ones, C * sizeof(float), cudaMemcpyHostToDevice));
    NN_CUDA_CHECK(cudaMemset(st->beta, 0, C * sizeof(float)));
    free(ones);

    m->name = "groupnorm";
    m->in_dim = H * W * C;
    m->out_dim = H * W * C;
    m->state = st;
    m->forward = groupnorm_forward;
    m->backward = groupnorm_backward;
    m->parameters = groupnorm_parameters;
    m->free = groupnorm_free;
    return m;
}
