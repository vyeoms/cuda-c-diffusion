#ifndef SAMPLERS_HEUN_H
#define SAMPLERS_HEUN_H

#include "denoiser.h"

/* Heun (2nd-order) sampler for the probability-flow ODE
     dx/dsigma = (x - D(x, sigma)) / sigma
   over the EDM geometric sigma schedule (sigma_max -> sigma_min -> 0). Writes
   `batch` samples into `x` (device); `x` is seeded with N(0, sigma_max^2)
   internally. Allocates its own scratch. */
void heun_sample(Denoiser* den, Context* ctx, float* x, int batch,
    float sigma_max, float sigma_min, int steps, int verbose);

#endif /* SAMPLERS_HEUN_H */
