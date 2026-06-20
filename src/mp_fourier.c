#include "nn.h"
#include <stdlib.h>

typedef struct {
    int C;
    float *freqs, *phases;
} FourierState;

static void mp_fourier_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    FourierState* st = (FourierState*)self->state;
    launch_mp_fourier(out, in, st->freqs, st->phases, batch, st->C, ctx->stream);
}

static void mp_fourier_backward(Module* self, const float* in, const float* out,
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
    /* fixed frequencies/phases and a data-only input: no gradient flows here */
}

static void mp_fourier_free(Module* self)
{
    FourierState* st = (FourierState*)self->state;
    cudaFree(st->freqs);
    cudaFree(st->phases);
    free(st);
    free(self);
}

Module* mp_fourier_create(Context* ctx, int num_channels, float bandwidth)
{
    Module* m = (Module*)malloc(sizeof(Module));
    FourierState* st = (FourierState*)malloc(sizeof(FourierState));
    st->C = num_channels;
    st->freqs = nn_device_alloc(num_channels);
    st->phases = nn_device_alloc(num_channels);

    /* freqs ~ N(0, bandwidth);  phases ~ U[0,1).  The kernel applies the 2*pi:
       out = sqrt(2)*cos(2*pi*(x*freq + phase)), matching EDM2's MPFourier. */
    nn_fill_normal(ctx->curand, st->freqs, num_channels, 0.0f, bandwidth);
    NN_CURAND_CHECK(curandGenerateUniform(ctx->curand, st->phases, num_channels));

    m->name = "mp_fourier";
    m->in_dim = 1;
    m->out_dim = num_channels;
    m->state = st;
    m->forward = mp_fourier_forward;
    m->backward = mp_fourier_backward;
    m->parameters = NULL;
    m->free = mp_fourier_free;
    return m;
}
