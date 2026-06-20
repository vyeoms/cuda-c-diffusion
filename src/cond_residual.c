#include "nn.h"
#include <math.h>
#include <stdlib.h>

/* EDM2 Block (no attention):
     h0 = mp_silu(x)
     h1 = lin0(h0)
     c  = emb_lin(emb)            (zero-init gain -> starts at 0)
     h2 = h1 * (c + 1)            (FiLM modulation)
     h3 = mp_silu(h2)
     h4 = lin1(h3)
     out = mp_sum(x, h4, t)                                                   */

typedef struct {
    int dim, emb_dim, max_batch, cond_slot;
    float wa, wb;
    Module *lin0, *emb_lin, *lin1;
    float *h0, *h1, *c, *h2, *h3, *h4;
    float *gh4, *gh3, *gh2, *gh1, *gh0, *gc, *gxb, *grad_emb_local;
} CondResState;

static void crb_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    CondResState* st = (CondResState*)self->state;
    int n = batch * st->dim;
    launch_activation_forward(st->h0, in, n, ACT_MP_SILU, ctx->stream);
    st->lin0->forward(st->lin0, st->h0, st->h1, batch, ctx);
    st->emb_lin->forward(st->emb_lin, ctx->cond[st->cond_slot].fwd, st->c, batch, ctx);
    launch_modulate(st->h2, st->h1, st->c, n, ctx->stream);
    launch_activation_forward(st->h3, st->h2, n, ACT_MP_SILU, ctx->stream);
    st->lin1->forward(st->lin1, st->h3, st->h4, batch, ctx);
    launch_weighted_sum(out, in, st->h4, st->wa, st->wb, n, ctx->stream);
}

static void crb_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)out;
    CondResState* st = (CondResState*)self->state;
    int n = batch * st->dim;

    /* mp_sum backward: branch sees wb*grad_out (scaled for correct param grads) */
    launch_weighted_sum(st->gh4, grad_out, grad_out, st->wb, 0.0f, n, ctx->stream);
    st->lin1->backward(st->lin1, st->h3, st->h4, st->gh4, st->gh3, batch, ctx);
    launch_activation_backward(st->gh2, st->gh3, st->h2, n, ACT_MP_SILU, ctx->stream);

    /* modulation backward: gh1 = gh2*(c+1);  gc = gh2*h1 */
    launch_modulate_backward(st->gh1, st->gc, st->gh2, st->h1, st->c, n, ctx->stream);

    st->lin0->backward(st->lin0, st->h0, st->h1, st->gh1, st->gh0, batch, ctx);
    launch_activation_backward(st->gxb, st->gh0, in, n, ACT_MP_SILU, ctx->stream);

    st->emb_lin->backward(st->emb_lin, ctx->cond[st->cond_slot].fwd, st->c, st->gc,
        st->grad_emb_local, batch, ctx);
    launch_weighted_sum(ctx->cond[st->cond_slot].grad, ctx->cond[st->cond_slot].grad,
        st->grad_emb_local, 1.0f, 1.0f, batch * st->emb_dim, ctx->stream);

    launch_weighted_sum(grad_in, grad_out, st->gxb, st->wa, 1.0f, n, ctx->stream);
}

static void crb_parameters(Module* self, ParamList* pl)
{
    CondResState* st = (CondResState*)self->state;
    st->lin0->parameters(st->lin0, pl);
    st->emb_lin->parameters(st->emb_lin, pl);
    st->lin1->parameters(st->lin1, pl);
}

static void crb_free(Module* self)
{
    CondResState* st = (CondResState*)self->state;
    st->lin0->free(st->lin0);
    st->emb_lin->free(st->emb_lin);
    st->lin1->free(st->lin1);
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

Module* cond_residual_create(Context* ctx, int dim, int emb_dim, int max_batch, float t,
    int cond_slot)
{
    Module* m = (Module*)malloc(sizeof(Module));
    CondResState* st = (CondResState*)malloc(sizeof(CondResState));
    st->dim = dim;
    st->emb_dim = emb_dim;
    st->max_batch = max_batch;
    st->cond_slot = cond_slot;

    float d = sqrtf((1.0f - t) * (1.0f - t) + t * t);
    st->wa = (1.0f - t) / d;
    st->wb = t / d;

    st->lin0 = mp_linear_create(ctx, dim, dim, 1.0f);
    st->emb_lin = mp_linear_create(ctx, emb_dim, dim, 0.0f);
    st->lin1 = mp_linear_create(ctx, dim, dim, 1.0f);

    size_t nd = (size_t)max_batch * dim;
    st->h0 = nn_device_alloc(nd);
    st->h1 = nn_device_alloc(nd);
    st->c = nn_device_alloc(nd);
    st->h2 = nn_device_alloc(nd);
    st->h3 = nn_device_alloc(nd);
    st->h4 = nn_device_alloc(nd);
    st->gh4 = nn_device_alloc(nd);
    st->gh3 = nn_device_alloc(nd);
    st->gh2 = nn_device_alloc(nd);
    st->gh1 = nn_device_alloc(nd);
    st->gh0 = nn_device_alloc(nd);
    st->gc = nn_device_alloc(nd);
    st->gxb = nn_device_alloc(nd);
    st->grad_emb_local = nn_device_alloc((size_t)max_batch * emb_dim);

    m->name = "cond_residual";
    m->in_dim = dim;
    m->out_dim = dim;
    m->state = st;
    m->forward = crb_forward;
    m->backward = crb_backward;
    m->parameters = crb_parameters;
    m->free = crb_free;
    return m;
}
