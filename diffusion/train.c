#include "train.h"
#include <stdlib.h>
#include <sys/stat.h>

void train_diffusion(Denoiser* den, ParamList* params, Optimizer* opt, EMA* ema,
    void* loss, DiffusionLossStep loss_step,
    const float* h_images, int n_train, int img_dim, int max_batch,
    TrainConfig cfg, Context* ctx)
{
    float* d_clean = nn_device_alloc((size_t)max_batch * img_dim);

    int resuming = (cfg.resume_path && cfg.resume_path[0]);
    int checkpointing = (cfg.save_every > 0);

    /* Shadow/moment buffers are stable for the run, so list them once. */
    ParamList shadow, adam_m, adam_v;
    if (resuming || checkpointing) {
        param_list_init(&shadow);
        param_list_init(&adam_m);
        param_list_init(&adam_v);
        ema_shadow_list(ema, &shadow);
        optimizer_moment_lists(opt, &adam_m, &adam_v);
    }
    if (checkpointing)
        mkdir(cfg.ckpt_dir, 0755);

    int start = 1;
    if (resuming) {
        int loaded_step = 0, adam_t = 0;
        int rc = checkpoint_load_training(cfg.resume_path, &loaded_step, &adam_t,
            params, &shadow, &adam_m, &adam_v);
        NN_ASSERT(rc == 0, "train: resume checkpoint load failed");
        optimizer_set_timestep(opt, adam_t);
        start = loaded_step + 1;
        printf("resumed from %s at step %d\n", cfg.resume_path, loaded_step);
    }

    printf("\n=== Training (steps %d..%d, batch=%d) ===\n",
        start, cfg.train_steps, max_batch);
    for (int step = start; step <= cfg.train_steps; ++step) {
        float cur_lr = lr_inv_sqrt(cfg.lr, step, cfg.lr_warmup, cfg.lr_ref);
        optimizer_set_lr(opt, cur_lr);

        for (int b = 0; b < max_batch; ++b) {
            int idx = rand() % n_train;
            NN_CUDA_CHECK(cudaMemcpy(d_clean + (size_t)b * img_dim,
                h_images + (size_t)idx * img_dim,
                img_dim * sizeof(float), cudaMemcpyHostToDevice));
        }

        float loss_val = loss_step(loss, den, d_clean, max_batch, ctx);

        optimizer_step(opt, params, ctx);
        ema_update(ema, params, ctx);

        if (step == start || step % cfg.log_every == 0)
            printf("step %5d   %s %.6f   lr %.2e\n",
                step, cfg.loss_label, loss_val, cur_lr);

        if (checkpointing && step % cfg.save_every == 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s/step_%d.ckpt", cfg.ckpt_dir, step);
            NN_CUDA_CHECK(cudaDeviceSynchronize());
            checkpoint_save_training(path, step, optimizer_timestep(opt),
                params, &shadow, &adam_m, &adam_v);
            printf("  checkpoint -> %s\n", path);
        }
    }

    if (resuming || checkpointing) {
        param_list_free(&shadow);
        param_list_free(&adam_m);
        param_list_free(&adam_v);
    }
    cudaFree(d_clean);
}
