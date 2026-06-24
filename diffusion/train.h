#ifndef DIFFUSION_TRAIN_H
#define DIFFUSION_TRAIN_H

#include "denoiser.h"

/* A framework's training step: noise d_x0 at random levels, run the denoiser
   forward+backward, return the loss. `loss` is the framework's loss object as
   void* (the void*-state idiom used by Module and Denoiser). */
typedef float (*DiffusionLossStep)(void* loss, Denoiser* den,
    const float* d_x0, int batch, Context* ctx);

typedef struct {
    int train_steps;
    float lr;
    int lr_warmup, lr_ref;
    int log_every;
    const char* loss_label; /* logged column name, e.g. "mse" / "wmse" */
    int save_every; /* 0 disables periodic checkpointing */
    const char* ckpt_dir; /* dir for step_<n>.ckpt (created on demand) */
    const char* resume_path; /* if non-empty, load full training state first */
} TrainConfig;

/* Generic training loop (LR schedule, batch staging, loss step, optimizer/EMA,
   logging, periodic checkpoint/resume). params/opt/ema stay caller-owned -- the
   caller needs them for sampling and cleanup. */
void train_diffusion(Denoiser* den, ParamList* params, Optimizer* opt, EMA* ema,
    void* loss, DiffusionLossStep loss_step,
    const float* h_images, int n_train, int img_dim, int max_batch,
    TrainConfig cfg, Context* ctx);

#endif /* DIFFUSION_TRAIN_H */
