#ifndef DIFFUSION_DDPM_H
#define DIFFUSION_DDPM_H

#include "denoiser.h"

/* DDPM framework (x0-prediction): the denoiser is a GroupNorm UNet + sinusoidal
   timestep embedding, output directly the x0 estimate (no EDM preconditioning);
   level = timestep index. Loss is x0 MSE with min-SNR-gamma weighting; the
   schedule is the shared forward process. */

/* Linear-beta noise schedule (host-side arrays, t = 0..T-1). */
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

/* Build a DDPM denoiser. `unet_cfg` supplies channels/emb_dim/heads/num_groups;
   its cond_slot is assigned internally (the denoiser owns conditioning). */
Denoiser* ddpm_denoiser_create(Context* ctx, const DDPMUNetConfig* unet_cfg,
    int C, int H, int W, int max_batch);

/* ----------------------------- training loss ----------------------------- */

typedef struct DDPMLoss DDPMLoss;

DDPMLoss* ddpm_loss_create(const DDPMSchedule* sched, float snr_gamma,
    int img_dim, int max_batch);

/* Draws per-sample timestep, noises d_x0 via the schedule, runs the denoiser
   forward+backward, and returns the mean min-SNR-weighted MSE. `loss` is the
   DDPMLoss* (void* to match DiffusionLossStep). */
float ddpm_loss_step(void* loss, Denoiser* den, const float* d_x0,
    int batch, Context* ctx);

void ddpm_loss_free(DDPMLoss* l);

#endif /* DIFFUSION_DDPM_H */
