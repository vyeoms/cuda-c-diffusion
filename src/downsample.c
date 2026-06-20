#include "nn.h"
#include <stdlib.h>

typedef struct {
    int C, H, W, max_batch;
} DownState;

static void down_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    DownState* st = (DownState*)self->state;
    launch_avg_pool_2x(out, in, batch, st->H, st->W, st->C, ctx->stream);
}

static void down_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)in;
    (void)out;
    DownState* st = (DownState*)self->state;
    launch_avg_pool_2x_backward(grad_in, grad_out, batch, st->H, st->W, st->C,
        ctx->stream);
}

static void down_parameters(Module* self, ParamList* pl)
{
    (void)self;
    (void)pl;
}

static void down_free(Module* self)
{
    free(self->state);
    free(self);
}

Module* downsample_create(int C, int H, int W, int max_batch)
{
    Module* m = (Module*)malloc(sizeof(Module));
    DownState* st = (DownState*)malloc(sizeof(DownState));
    st->C = C;
    st->H = H;
    st->W = W;
    st->max_batch = max_batch;

    m->name = "downsample_2x";
    m->in_dim = H * W * C;
    m->out_dim = (H / 2) * (W / 2) * C;
    m->state = st;
    m->forward = down_forward;
    m->backward = down_backward;
    m->parameters = down_parameters;
    m->free = down_free;
    return m;
}
