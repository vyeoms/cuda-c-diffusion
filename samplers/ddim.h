#ifndef SAMPLERS_DDIM_H
#define SAMPLERS_DDIM_H

#include "ddpm.h" /* DDPMSchedule */
#include "denoiser.h"

/* Deterministic DDIM sampler (eta = 0) for an x0-prediction denoiser. */

typedef struct DDIMChain DDIMChain;

DDIMChain* ddim_chain_create(const DDPMSchedule* sched, int infer_steps,
    int img_dim, int max_batch);

/* x_T -> x_0 through `infer_steps` DDIM steps. If x_T is NULL, samples N(0,I). */
void ddim_chain_forward(DDIMChain* chain, Denoiser* den,
    const float* x_T, int batch, Context* ctx);

/* Pointer to the final generated images (device, batch * img_dim). */
float* ddim_chain_output(DDIMChain* chain);

void ddim_chain_free(DDIMChain* chain);

#endif /* SAMPLERS_DDIM_H */
