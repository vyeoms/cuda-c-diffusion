#include "gemm.h"
#include "nn.h"
#include <math.h>
#include <stdlib.h>

typedef struct {
    int C_in, C_out, K, stride, pad;
    int Hi, Wi, Ho, Wo;
    int max_batch, mp_mode;
    float eps;
    float *Wt, *gW;
    float *b, *gb;
    float *gain, *g_gain;
    float* col;
} Conv2DState;

static void conv2d_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    Conv2DState* st = (Conv2DState*)self->state;
    int M = batch * st->Ho * st->Wo;
    int Kg = st->K * st->K * st->C_in;

    launch_im2col(st->col, in, batch, st->Hi, st->Wi, st->C_in,
        st->K, st->stride, st->pad, st->Ho, st->Wo, ctx->stream);

    if (st->mp_mode) {
        launch_weight_normalize(st->Wt, Kg, st->C_out, st->eps, ctx->stream);
        gemm_forward(ctx->cublas, 1.0f, st->col, st->Wt, out, M, Kg, st->C_out);
        launch_scale_by_scalar(out, st->gain, M * st->C_out, ctx->stream);
    } else {
        gemm_forward(ctx->cublas, 1.0f, st->col, st->Wt, out, M, Kg, st->C_out);
        launch_add_bias(out, st->b, M, st->C_out, ctx->stream);
    }
}

static void conv2d_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)out;
    Conv2DState* st = (Conv2DState*)self->state;
    int M = batch * st->Ho * st->Wo;
    int Kg = st->K * st->K * st->C_in;

    launch_im2col(st->col, in, batch, st->Hi, st->Wi, st->C_in,
        st->K, st->stride, st->pad, st->Ho, st->Wo, ctx->stream);
    gemm_dW(ctx->cublas, 1.0f, st->col, grad_out, st->gW, M, Kg, st->C_out);

    if (st->mp_mode) {
        NN_CUDA_CHECK(cudaMemsetAsync(st->g_gain, 0, sizeof(float), ctx->stream));
        launch_weight_normalize_backward(st->gW, st->Wt, st->gain, st->g_gain,
            Kg, st->C_out, ctx->stream);
        gemm_dX(ctx->cublas, 1.0f, st->Wt, grad_out, st->col, M, Kg, st->C_out);
        launch_scale_by_scalar(st->col, st->gain, M * Kg, ctx->stream);
    } else {
        launch_bias_grad(st->gb, grad_out, M, st->C_out, ctx->stream);
        gemm_dX(ctx->cublas, 1.0f, st->Wt, grad_out, st->col, M, Kg, st->C_out);
    }

    NN_CUDA_CHECK(cudaMemsetAsync(grad_in, 0,
        (size_t)batch * st->Hi * st->Wi * st->C_in * sizeof(float),
        ctx->stream));
    launch_col2im(grad_in, st->col, batch, st->Hi, st->Wi, st->C_in,
        st->K, st->stride, st->pad, st->Ho, st->Wo, ctx->stream);
}

static void conv2d_parameters(Module* self, ParamList* pl)
{
    Conv2DState* st = (Conv2DState*)self->state;
    param_list_add(pl, st->Wt, st->gW, st->K * st->K * st->C_in * st->C_out);
    if (st->mp_mode)
        param_list_add(pl, st->gain, st->g_gain, 1);
    else
        param_list_add(pl, st->b, st->gb, st->C_out);
}

static void conv2d_free(Module* self)
{
    Conv2DState* st = (Conv2DState*)self->state;
    cudaFree(st->Wt);
    cudaFree(st->gW);
    cudaFree(st->col);
    if (st->mp_mode) {
        cudaFree(st->gain);
        cudaFree(st->g_gain);
    } else {
        cudaFree(st->b);
        cudaFree(st->gb);
    }
    free(st);
    free(self);
}

Module* conv2d_create(Context* ctx, int C_in, int C_out, int K, int stride, int pad,
    int H, int W, int max_batch, int mp_mode, float gain)
{
    Module* m = (Module*)malloc(sizeof(Module));
    Conv2DState* st = (Conv2DState*)malloc(sizeof(Conv2DState));

    st->C_in = C_in;
    st->C_out = C_out;
    st->K = K;
    st->stride = stride;
    st->pad = pad;
    st->Hi = H;
    st->Wi = W;
    st->Ho = (H + 2 * pad - K) / stride + 1;
    st->Wo = (W + 2 * pad - K) / stride + 1;
    st->max_batch = max_batch;
    st->mp_mode = mp_mode;
    st->eps = 1e-4f;

    int Kg = K * K * C_in;
    st->Wt = nn_device_alloc((size_t)Kg * C_out);
    st->gW = nn_device_alloc((size_t)Kg * C_out);
    st->col = nn_device_alloc((size_t)max_batch * st->Ho * st->Wo * Kg);

    if (mp_mode) {
        st->b = NULL;
        st->gb = NULL;
        st->gain = nn_device_alloc(1);
        st->g_gain = nn_device_alloc(1);
        nn_fill_normal(ctx->curand, st->Wt, (size_t)Kg * C_out, 0.0f, 1.0f);
        NN_CUDA_CHECK(cudaMemcpy(st->gain, &gain, sizeof(float), cudaMemcpyHostToDevice));
    } else {
        st->gain = NULL;
        st->g_gain = NULL;
        st->b = nn_device_alloc((size_t)C_out);
        st->gb = nn_device_alloc((size_t)C_out);
        float std = sqrtf(2.0f / (float)Kg);
        nn_fill_normal(ctx->curand, st->Wt, (size_t)Kg * C_out, 0.0f, std);
        NN_CUDA_CHECK(cudaMemset(st->b, 0, C_out * sizeof(float)));
    }

    m->name = "conv2d";
    m->in_dim = H * W * C_in;
    m->out_dim = st->Ho * st->Wo * C_out;
    m->state = st;
    m->forward = conv2d_forward;
    m->backward = conv2d_backward;
    m->parameters = conv2d_parameters;
    m->free = conv2d_free;
    return m;
}
