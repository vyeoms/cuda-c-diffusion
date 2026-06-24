#ifndef DIFFUSION_EDM_H
#define DIFFUSION_EDM_H

#include "denoiser.h"

/* EDM framework: denoiser (Karras preconditioner D = c_skip*x + c_out*F) and its
   loss, co-located because the loss defines the training dynamics. The denoiser
   owns the UNet, the Fourier time embedding, and its conditioning slot. */
Denoiser* edm_denoiser_create(Context* ctx, const UNetConfig* unet_cfg,
    int C, int H, int W, int max_batch, float sigma_data);

/* ----------------------------- training loss ----------------------------- */

typedef struct {
    float p_mean, p_std; /* log-sigma sampling distribution */
    float sigma_min, sigma_max; /* clamp range */
    float sigma_data;
} EDMLossCfg;

typedef struct EDMLoss EDMLoss;

EDMLoss* edm_loss_create(int img_dim, int max_batch, EDMLossCfg cfg);

/* Draws per-sample sigma, noises d_x0, runs the denoiser forward+backward, and
   returns the mean EDM-weighted MSE. `loss` is the EDMLoss* (void* to match
   DiffusionLossStep). */
float edm_loss_step(void* loss, Denoiser* den, const float* d_x0,
    int batch, Context* ctx);

void edm_loss_free(EDMLoss* l);

#endif /* DIFFUSION_EDM_H */
