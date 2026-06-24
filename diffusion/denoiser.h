#ifndef DIFFUSION_DENOISER_H
#define DIFFUSION_DENOISER_H

#include "nn.h"
#include <stdlib.h>

/* Maps a noisy input to an x0 estimate at a given noise level -- the one
   abstraction every sampler and loss shares (EDM: preconditioned D; x0-prediction
   DDPM: the network output). `levels` is per-sample (host, length batch) since
   training draws a level per example; samplers pass a uniform array below. */
typedef struct Denoiser Denoiser;
struct Denoiser {
    void* state;
    int img_dim;

    void (*forward)(Denoiser* self, Context* ctx, const float* x, float* out,
        int batch, const float* levels);

    /* Must follow forward() with the same x/levels. No input gradient: the input
       is data, not a trainable leaf. */
    void (*backward)(Denoiser* self, Context* ctx, const float* x,
        const float* grad_out, int batch, const float* levels);

    void (*parameters)(Denoiser* self, ParamList* pl);
    void (*free)(Denoiser* self);
};
static inline void denoiser_forward_uniform(Denoiser* d, Context* ctx,
    const float* x, float* out, int batch, float level)
{
    float stackbuf[256] = { 0 };
    float* lv = (batch <= 256) ? stackbuf : (float*)malloc((size_t)batch * sizeof(float));
    for (int b = 0; b < batch; ++b)
        lv[b] = level;
    d->forward(d, ctx, x, out, batch, lv);
    if (lv != stackbuf)
        free(lv);
}

#endif /* DIFFUSION_DENOISER_H */
