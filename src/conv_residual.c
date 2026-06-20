#include "nn.h"
#include <stdlib.h>

/* Classic DDPM ResBlock:
     h = GroupNorm(x) → SiLU → Conv3x3
     h += Linear(emb)           (additive conditioning, broadcast over H,W)
     h = GroupNorm(h) → SiLU → Conv3x3
     out = x + h                                                              */

typedef struct {
    int C, emb_dim, max_batch, H, W, G, cond_slot;
    Module *gn1, *gn2;
    Module *conv1, *conv2;
    Module* time_proj;
    float *h0, *h1, *h2, *h3, *h4;
    float* c;
    float *gh4, *gh3, *gh2, *gh1, *gh0, *gxb;
    float* gc;
    float* grad_emb_local;
} ConvResState;

static void cr_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    ConvResState* st = (ConvResState*)self->state;
    int n = batch * st->H * st->W * st->C;

    st->gn1->forward(st->gn1, in, st->h0, batch, ctx);
    launch_activation_forward(st->h1, st->h0, n, ACT_SILU, ctx->stream);
    st->conv1->forward(st->conv1, st->h1, st->h2, batch, ctx);

    st->time_proj->forward(st->time_proj, ctx->cond[st->cond_slot].fwd, st->c, batch, ctx);
    launch_spatial_add(st->h2, st->c, batch, st->H, st->W, st->C, ctx->stream);

    st->gn2->forward(st->gn2, st->h2, st->h3, batch, ctx);
    launch_activation_forward(st->h4, st->h3, n, ACT_SILU, ctx->stream);
    st->conv2->forward(st->conv2, st->h4, out, batch, ctx);

    launch_weighted_sum(out, in, out, 1.0f, 1.0f, n, ctx->stream);
}

static void cr_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)out;
    ConvResState* st = (ConvResState*)self->state;
    int n = batch * st->H * st->W * st->C;

    st->conv2->backward(st->conv2, st->h4, NULL, grad_out, st->gh4, batch, ctx);
    launch_activation_backward(st->gh3, st->gh4, st->h3, n, ACT_SILU, ctx->stream);
    st->gn2->backward(st->gn2, st->h2, NULL, st->gh3, st->gh2, batch, ctx);

    launch_spatial_reduce_sum(st->gc, st->gh2, batch, st->H * st->W, st->C,
        ctx->stream);
    st->time_proj->backward(st->time_proj, ctx->cond[st->cond_slot].fwd, st->c, st->gc,
        st->grad_emb_local, batch, ctx);
    launch_weighted_sum(ctx->cond[st->cond_slot].grad, ctx->cond[st->cond_slot].grad,
        st->grad_emb_local, 1.0f, 1.0f, batch * st->emb_dim, ctx->stream);

    st->conv1->backward(st->conv1, st->h1, NULL, st->gh2, st->gh1, batch, ctx);
    launch_activation_backward(st->gh0, st->gh1, st->h0, n, ACT_SILU, ctx->stream);
    st->gn1->backward(st->gn1, in, NULL, st->gh0, st->gxb, batch, ctx);

    launch_weighted_sum(grad_in, grad_out, st->gxb, 1.0f, 1.0f, n, ctx->stream);
}

static void cr_parameters(Module* self, ParamList* pl)
{
    ConvResState* st = (ConvResState*)self->state;
    st->gn1->parameters(st->gn1, pl);
    st->conv1->parameters(st->conv1, pl);
    st->time_proj->parameters(st->time_proj, pl);
    st->gn2->parameters(st->gn2, pl);
    st->conv2->parameters(st->conv2, pl);
}

static void cr_free(Module* self)
{
    ConvResState* st = (ConvResState*)self->state;
    st->gn1->free(st->gn1);
    st->gn2->free(st->gn2);
    st->conv1->free(st->conv1);
    st->conv2->free(st->conv2);
    st->time_proj->free(st->time_proj);
    cudaFree(st->h0);
    cudaFree(st->h1);
    cudaFree(st->h2);
    cudaFree(st->h3);
    cudaFree(st->h4);
    cudaFree(st->c);
    cudaFree(st->gh4);
    cudaFree(st->gh3);
    cudaFree(st->gh2);
    cudaFree(st->gh1);
    cudaFree(st->gh0);
    cudaFree(st->gxb);
    cudaFree(st->gc);
    cudaFree(st->grad_emb_local);
    free(st);
    free(self);
}

Module* conv_residual_create(Context* ctx, int C, int emb_dim, int H, int W,
    int max_batch, int num_groups, int cond_slot)
{
    Module* m = (Module*)malloc(sizeof(Module));
    ConvResState* st = (ConvResState*)malloc(sizeof(ConvResState));
    st->C = C;
    st->emb_dim = emb_dim;
    st->max_batch = max_batch;
    st->H = H;
    st->W = W;
    st->G = num_groups;
    st->cond_slot = cond_slot;

    st->gn1 = spatial_groupnorm_create(C, num_groups, H, W, max_batch);
    st->conv1 = conv2d_create(ctx, C, C, 3, 1, 1, H, W, max_batch, 0, 1.0f);
    st->time_proj = linear_create(ctx, emb_dim, C);
    st->gn2 = spatial_groupnorm_create(C, num_groups, H, W, max_batch);
    st->conv2 = conv2d_create(ctx, C, C, 3, 1, 1, H, W, max_batch, 0, 1.0f);

    size_t ns = (size_t)max_batch * H * W * C;
    st->h0 = nn_device_alloc(ns);
    st->h1 = nn_device_alloc(ns);
    st->h2 = nn_device_alloc(ns);
    st->h3 = nn_device_alloc(ns);
    st->h4 = nn_device_alloc(ns);
    st->c = nn_device_alloc((size_t)max_batch * C);
    st->gh4 = nn_device_alloc(ns);
    st->gh3 = nn_device_alloc(ns);
    st->gh2 = nn_device_alloc(ns);
    st->gh1 = nn_device_alloc(ns);
    st->gh0 = nn_device_alloc(ns);
    st->gxb = nn_device_alloc(ns);
    st->gc = nn_device_alloc((size_t)max_batch * C);
    st->grad_emb_local = nn_device_alloc((size_t)max_batch * emb_dim);

    int dim = H * W * C;
    m->name = "conv_residual";
    m->in_dim = dim;
    m->out_dim = dim;
    m->state = st;
    m->forward = cr_forward;
    m->backward = cr_backward;
    m->parameters = cr_parameters;
    m->free = cr_free;
    return m;
}
