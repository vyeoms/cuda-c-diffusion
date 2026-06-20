#include "nn.h"
#include <stdlib.h>

typedef struct {
    int C, H, W, max_batch;
} UpState;

static void up_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    UpState* st = (UpState*)self->state;
    launch_nearest_upsample_2x(out, in, batch, st->H, st->W, st->C, ctx->stream);
}

static void up_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)in;
    (void)out;
    UpState* st = (UpState*)self->state;
    launch_nearest_upsample_2x_backward(grad_in, grad_out, batch, st->H, st->W, st->C,
        ctx->stream);
}

static void up_parameters(Module* self, ParamList* pl)
{
    (void)self;
    (void)pl;
}

static void up_free(Module* self)
{
    free(self->state);
    free(self);
}

Module* upsample_create(int C, int H, int W, int max_batch)
{
    Module* m = (Module*)malloc(sizeof(Module));
    UpState* st = (UpState*)malloc(sizeof(UpState));
    st->C = C;
    st->H = H;
    st->W = W;
    st->max_batch = max_batch;

    m->name = "upsample_2x";
    m->in_dim = H * W * C;
    m->out_dim = (2 * H) * (2 * W) * C;
    m->state = st;
    m->forward = up_forward;
    m->backward = up_backward;
    m->parameters = up_parameters;
    m->free = up_free;
    return m;
}
