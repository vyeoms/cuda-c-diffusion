#include "gemm.h"
#include "nn.h"
#include <stdlib.h>

typedef struct {
    int in, out;
    float eps;
    float *W, *gW; /* unit-normalized weights; MP layers carry no bias */
    float *gain, *g_gain;
} MPLinearState;

static void mp_linear_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    MPLinearState* st = (MPLinearState*)self->state;
    launch_weight_normalize(st->W, st->in, st->out, st->eps, ctx->stream);
    gemm_forward(ctx->cublas, 1.0f, in, st->W, out, batch, st->in, st->out);
    launch_scale_by_scalar(out, st->gain, batch * st->out, ctx->stream);
}

static void mp_linear_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)out;
    MPLinearState* st = (MPLinearState*)self->state;

    /* gW <- M = x^T @ grad_out  (gain not yet applied) */
    gemm_dW(ctx->cublas, 1.0f, in, grad_out, st->gW, batch, st->in, st->out);

    NN_CUDA_CHECK(cudaMemsetAsync(st->g_gain, 0, sizeof(float), ctx->stream));
    launch_weight_normalize_backward(st->gW, st->W, st->gain, st->g_gain,
        st->in, st->out, ctx->stream);

    /* dL/dx = gain * (grad_out @ Wn^T) */
    if (grad_in) {
        gemm_dX(ctx->cublas, 1.0f, st->W, grad_out, grad_in, batch, st->in, st->out);
        launch_scale_by_scalar(grad_in, st->gain, batch * st->in, ctx->stream);
    }
}

static void mp_linear_parameters(Module* self, ParamList* pl)
{
    MPLinearState* st = (MPLinearState*)self->state;
    param_list_add(pl, st->W, st->gW, st->in * st->out);
    param_list_add(pl, st->gain, st->g_gain, 1);
}

static void mp_linear_free(Module* self)
{
    MPLinearState* st = (MPLinearState*)self->state;
    cudaFree(st->W);
    cudaFree(st->gW);
    cudaFree(st->gain);
    cudaFree(st->g_gain);
    free(st);
    free(self);
}

Module* mp_linear_create(Context* ctx, int in_dim, int out_dim, float gain_init)
{
    Module* m = (Module*)malloc(sizeof(Module));
    MPLinearState* st = (MPLinearState*)malloc(sizeof(MPLinearState));
    st->in = in_dim;
    st->out = out_dim;
    st->eps = 1e-4f;
    st->W = nn_device_alloc((size_t)in_dim * out_dim);
    st->gW = nn_device_alloc((size_t)in_dim * out_dim);
    st->gain = nn_device_alloc(1);
    st->g_gain = nn_device_alloc(1);

    /* Gaussian init -> isotropic directions on the unit sphere after weight-norm */
    nn_fill_normal(ctx->curand, st->W, (size_t)in_dim * out_dim, 0.0f, 1.0f);

    NN_CUDA_CHECK(cudaMemcpy(st->gain, &gain_init, sizeof(float), cudaMemcpyHostToDevice));

    m->name = "mp_linear";
    m->in_dim = in_dim;
    m->out_dim = out_dim;
    m->state = st;
    m->forward = mp_linear_forward;
    m->backward = mp_linear_backward;
    m->parameters = mp_linear_parameters;
    m->free = mp_linear_free;
    return m;
}
