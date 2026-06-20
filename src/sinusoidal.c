#include "nn.h"
#include <math.h>
#include <stdlib.h>

typedef struct {
    int C;
    float log_max_period;
} SinusoidalState;

static void sinusoidal_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    SinusoidalState* st = (SinusoidalState*)self->state;
    launch_sinusoidal(out, in, batch, st->C, st->log_max_period, ctx->stream);
}

static void sinusoidal_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)self;
    (void)in;
    (void)out;
    (void)grad_out;
    (void)grad_in;
    (void)batch;
    (void)ctx;
    /* deterministic, data-only input: no gradient */
}

static void sinusoidal_free(Module* self)
{
    free(self->state);
    free(self);
}

Module* sinusoidal_create(int num_channels, float max_period)
{
    if (num_channels % 2 != 0) {
        fprintf(stderr, "sinusoidal_create: num_channels must be even (got %d)\n",
            num_channels);
        exit(1);
    }
    Module* m = (Module*)malloc(sizeof(Module));
    SinusoidalState* st = (SinusoidalState*)malloc(sizeof(SinusoidalState));
    st->C = num_channels;
    st->log_max_period = logf(max_period);

    m->name = "sinusoidal";
    m->in_dim = 1;
    m->out_dim = num_channels;
    m->state = st;
    m->forward = sinusoidal_forward;
    m->backward = sinusoidal_backward;
    m->parameters = NULL;
    m->free = sinusoidal_free;
    return m;
}
