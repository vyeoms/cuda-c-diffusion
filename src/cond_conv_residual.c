#include "nn.h"
#include <math.h>
#include <stdlib.h>

/* EDM2 Block (spatial convolution variant):
     h0 = mp_silu(x)
     h1 = conv0(h0)              (K=3, pad=1, stride=1 -> same spatial dims)
     c  = emb_lin(emb)           (zero-init gain -> starts at 0)
     h2 = h1 * (c + 1)           (spatial FiLM: c[b,C] broadcast over H,W)
     h3 = mp_silu(h2)
     h4 = conv1(h3)
     out = mp_sum(x, h4, t)                                                    */

typedef struct {
    int C, emb_dim, max_batch, H, W, cond_slot;
    float wa, wb;
    Module *conv0, *emb_lin, *conv1;
    float *h0, *h1, *c, *h2, *h3, *h4;
    float *gh4, *gh3, *gh2, *gh1, *gh0, *gc, *gxb, *grad_emb_local;
} CondConvResState;

static void ccr_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    CondConvResState* st = (CondConvResState*)self->state;
    int n = batch * st->H * st->W * st->C;

    launch_activation_forward(st->h0, in, n, ACT_MP_SILU, ctx->stream);
    st->conv0->forward(st->conv0, st->h0, st->h1, batch, ctx);
    st->emb_lin->forward(st->emb_lin, ctx->cond[st->cond_slot].fwd, st->c, batch, ctx);
    launch_spatial_modulate(st->h2, st->h1, st->c,
        batch, st->H, st->W, st->C, ctx->stream);
    launch_activation_forward(st->h3, st->h2, n, ACT_MP_SILU, ctx->stream);
    st->conv1->forward(st->conv1, st->h3, st->h4, batch, ctx);
    launch_weighted_sum(out, in, st->h4, st->wa, st->wb, n, ctx->stream);
}

static void ccr_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)out;
    CondConvResState* st = (CondConvResState*)self->state;
    int n = batch * st->H * st->W * st->C;

    launch_weighted_sum(st->gh4, grad_out, grad_out, st->wb, 0.0f, n, ctx->stream);
    st->conv1->backward(st->conv1, st->h3, st->h4, st->gh4, st->gh3, batch, ctx);
    launch_activation_backward(st->gh2, st->gh3, st->h2, n, ACT_MP_SILU, ctx->stream);

    launch_spatial_modulate_backward(st->gh1, st->gc, st->gh2, st->h1, st->c,
        batch, st->H, st->W, st->C, ctx->stream);

    st->conv0->backward(st->conv0, st->h0, st->h1, st->gh1, st->gh0, batch, ctx);
    launch_activation_backward(st->gxb, st->gh0, in, n, ACT_MP_SILU, ctx->stream);

    st->emb_lin->backward(st->emb_lin, ctx->cond[st->cond_slot].fwd, st->c, st->gc,
        st->grad_emb_local, batch, ctx);
    launch_weighted_sum(ctx->cond[st->cond_slot].grad, ctx->cond[st->cond_slot].grad,
        st->grad_emb_local, 1.0f, 1.0f, batch * st->emb_dim, ctx->stream);

    launch_weighted_sum(grad_in, grad_out, st->gxb, st->wa, 1.0f, n, ctx->stream);
}

static void ccr_parameters(Module* self, ParamList* pl)
{
    CondConvResState* st = (CondConvResState*)self->state;
    st->conv0->parameters(st->conv0, pl);
    st->emb_lin->parameters(st->emb_lin, pl);
    st->conv1->parameters(st->conv1, pl);
}

static void ccr_free(Module* self)
{
    CondConvResState* st = (CondConvResState*)self->state;
    st->conv0->free(st->conv0);
    st->emb_lin->free(st->emb_lin);
    st->conv1->free(st->conv1);
    cudaFree(st->h0);
    cudaFree(st->h1);
    cudaFree(st->c);
    cudaFree(st->h2);
    cudaFree(st->h3);
    cudaFree(st->h4);
    cudaFree(st->gh4);
    cudaFree(st->gh3);
    cudaFree(st->gh2);
    cudaFree(st->gh1);
    cudaFree(st->gh0);
    cudaFree(st->gc);
    cudaFree(st->gxb);
    cudaFree(st->grad_emb_local);
    free(st);
    free(self);
}

Module* cond_conv_residual_create(Context* ctx, int C, int emb_dim, int H, int W,
    int max_batch, float t, int cond_slot)
{
    Module* m = (Module*)malloc(sizeof(Module));
    CondConvResState* st = (CondConvResState*)malloc(sizeof(CondConvResState));
    st->C = C;
    st->emb_dim = emb_dim;
    st->max_batch = max_batch;
    st->H = H;
    st->W = W;
    st->cond_slot = cond_slot;

    float d = sqrtf((1.0f - t) * (1.0f - t) + t * t);
    st->wa = (1.0f - t) / d;
    st->wb = t / d;

    st->conv0 = conv2d_create(ctx, C, C, 3, 1, 1, H, W, max_batch, 1, 1.0f);
    st->emb_lin = mp_linear_create(ctx, emb_dim, C, 0.0f);
    st->conv1 = conv2d_create(ctx, C, C, 3, 1, 1, H, W, max_batch, 1, 1.0f);

    size_t ns = (size_t)max_batch * H * W * C;
    st->h0 = nn_device_alloc(ns);
    st->h1 = nn_device_alloc(ns);
    st->c = nn_device_alloc((size_t)max_batch * C);
    st->h2 = nn_device_alloc(ns);
    st->h3 = nn_device_alloc(ns);
    st->h4 = nn_device_alloc(ns);
    st->gh4 = nn_device_alloc(ns);
    st->gh3 = nn_device_alloc(ns);
    st->gh2 = nn_device_alloc(ns);
    st->gh1 = nn_device_alloc(ns);
    st->gh0 = nn_device_alloc(ns);
    st->gc = nn_device_alloc((size_t)max_batch * C);
    st->gxb = nn_device_alloc(ns);
    st->grad_emb_local = nn_device_alloc((size_t)max_batch * emb_dim);

    int dim = H * W * C;
    m->name = "cond_conv_residual";
    m->in_dim = dim;
    m->out_dim = dim;
    m->state = st;
    m->forward = ccr_forward;
    m->backward = ccr_backward;
    m->parameters = ccr_parameters;
    m->free = ccr_free;
    return m;
}
