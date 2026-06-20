#include "gemm.h"
#include "nn.h"
#include <math.h>

typedef struct {
    int in, out;
    float *W, *b; /* parameters (device) */
    float *gW, *gb; /* gradients  (device) */
} LinearState;

static void linear_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    LinearState* st = (LinearState*)self->state;
    gemm_forward(ctx->cublas, 1.0f, in, st->W, out, batch, st->in, st->out);
    launch_add_bias(out, st->b, batch, st->out, ctx->stream);
}

static void linear_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)out;
    LinearState* st = (LinearState*)self->state;
    gemm_dW(ctx->cublas, 1.0f, in, grad_out, st->gW, batch, st->in, st->out);
    launch_bias_grad(st->gb, grad_out, batch, st->out, ctx->stream);
    if (grad_in)
        gemm_dX(ctx->cublas, 1.0f, st->W, grad_out, grad_in, batch, st->in, st->out);
}

static void linear_parameters(Module* self, ParamList* pl)
{
    LinearState* st = (LinearState*)self->state;
    param_list_add(pl, st->W, st->gW, st->in * st->out);
    param_list_add(pl, st->b, st->gb, st->out);
}

static void linear_free(Module* self)
{
    LinearState* st = (LinearState*)self->state;
    cudaFree(st->W);
    cudaFree(st->b);
    cudaFree(st->gW);
    cudaFree(st->gb);
    free(st);
    free(self);
}

Module* linear_create(Context* ctx, int in_dim, int out_dim)
{
    Module* m = (Module*)malloc(sizeof(Module));
    LinearState* st = (LinearState*)malloc(sizeof(LinearState));
    st->in = in_dim;
    st->out = out_dim;
    st->W = nn_device_alloc((size_t)in_dim * out_dim);
    st->b = nn_device_alloc((size_t)out_dim);
    st->gW = nn_device_alloc((size_t)in_dim * out_dim);
    st->gb = nn_device_alloc((size_t)out_dim);

    /* He initialization: the sqrt(2/fan_in) scale is the generator's stddev */
    float std = sqrtf(2.0f / (float)in_dim);
    nn_fill_normal(ctx->curand, st->W, (size_t)in_dim * out_dim, 0.0f, std);
    NN_CUDA_CHECK(cudaMemset(st->b, 0, out_dim * sizeof(float)));

    m->name = "linear";
    m->in_dim = in_dim;
    m->out_dim = out_dim;
    m->state = st;
    m->forward = linear_forward;
    m->backward = linear_backward;
    m->parameters = linear_parameters;
    m->free = linear_free;
    return m;
}
