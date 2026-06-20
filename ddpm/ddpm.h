#ifndef DDPM_H
#define DDPM_H

#include "nn.h"

/* DDPM noise schedule (linear beta). All arrays host-side, 0-indexed: t = 0..T-1. */
typedef struct {
    int T;
    float* beta; /* [T] */
    float* alpha; /* [T]  1 - beta */
    float* alpha_bar; /* [T]  cumulative product of alpha */
    float* sqrt_ab; /* [T]  sqrt(alpha_bar) */
    float* sqrt_1mab; /* [T]  sqrt(1 - alpha_bar) */
} DDPMSchedule;

void ddpm_schedule_init(DDPMSchedule* s, int T, float beta_start, float beta_end);
void ddpm_schedule_free(DDPMSchedule* s);

typedef struct DDIMChain DDIMChain;

DDIMChain* ddim_chain_create(DDPMSchedule* sched, int infer_steps,
    int img_dim, int max_batch, int cond_slot);

/* Forward: x_T → x_0 through S DDIM steps. If x_T is NULL, samples N(0,I). */
void ddim_chain_forward(DDIMChain* chain, DDPMUNet* unet, Sequential* embed,
    const float* x_T, int batch, Context* ctx);

/* Pointer to the final generated images (device, batch * img_dim). */
float* ddim_chain_output(DDIMChain* chain);

void ddim_chain_free(DDIMChain* chain);

#endif /* DDPM_H */
