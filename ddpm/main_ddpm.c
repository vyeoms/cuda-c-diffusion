#include "ddpm.h"
#include "ini.h"
#include "mnist_io.h"
#include "nn.h"
#include "pgm_io.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

int main(int argc, char** argv)
{
    const char* ini_path = (argc > 1) ? argv[1] : "ddpm/ddpm.ini";
    Ini ini;
    if (ini_load(&ini, ini_path) != 0) {
        fprintf(stderr, "failed to load config: %s\n", ini_path);
        return 1;
    }
    printf("config: %s\n", ini_path);

    int max_batch = ini_int(&ini, "batch_size", 8);
    int cemb = ini_int(&ini, "emb_dim", 64);
    int train_steps = ini_int(&ini, "train_steps", 20000);
    int ddim_steps = ini_int(&ini, "ddim_steps", 50);
    float lr = ini_float(&ini, "lr", 1e-3f);
    int lr_warmup = ini_int(&ini, "lr_warmup", 500);
    int lr_ref = ini_int(&ini, "lr_ref", 2000);
    float snr_gamma = ini_float(&ini, "snr_gamma", 5.0f);
    int H = ini_int(&ini, "H", 28);
    int W = ini_int(&ini, "W", 28);
    int C_img = ini_int(&ini, "C", 1);
    int sched_T = ini_int(&ini, "T", 1000);
    float beta_start = ini_float(&ini, "beta_start", 1e-4f);
    float beta_end = ini_float(&ini, "beta_end", 0.02f);
    int heads = ini_int(&ini, "heads", 4);
    int num_groups = ini_int(&ini, "num_groups", 32);

    int channels[16];
    int n_ch = ini_int_list(&ini, "channels", channels, 16);
    if (n_ch < 2) {
        channels[0] = 32;
        channels[1] = 64;
        channels[2] = 128;
        n_ch = 3;
    }
    int levels = n_ch - 1;

    int img_dim = H * W * C_img;
    const char* data_path = ini_str(&ini, "images", "data/mnist_train_images.bin");

    /* ---- load data ---- */
    int n_train;
    float* h_images = mnist_load_images(data_path, &n_train);
    printf("loaded %d training images\n", n_train);

    /* ---- CUDA context ---- */
    Context ctx;
    NN_CUBLAS_CHECK(cublasCreate(&ctx.cublas));
    NN_CUBLAS_CHECK(cublasSetMathMode(ctx.cublas, CUBLAS_TF32_TENSOR_OP_MATH));
    ctx.stream = 0;
    NN_CURAND_CHECK(curandCreateGenerator(&ctx.curand, CURAND_RNG_PSEUDO_DEFAULT));
    NN_CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx.curand, 42ULL));
    NN_CURAND_CHECK(curandSetStream(ctx.curand, ctx.stream));
    ctx.n_cond = 0;
    int cond_slot = ctx_add_cond(&ctx, cemb, max_batch);

    /* ---- DDPM schedule ---- */
    DDPMSchedule sched;
    ddpm_schedule_init(&sched, sched_T, beta_start, beta_end);

    /* ---- precompute min-SNR-gamma weights ---- */
    float* snr_weight = (float*)malloc(sched_T * sizeof(float));
    for (int t = 0; t < sched_T; ++t) {
        float snr = sched.alpha_bar[t] / (1.0f - sched.alpha_bar[t]);
        snr_weight[t] = fminf(snr, snr_gamma) / snr_gamma;
    }

    /* ---- embedding pipeline: sinusoidal(t) → linear → silu → linear ---- */
    Module* emb_mods[] = {
        sinusoidal_create(cemb, 10000.0f),
        linear_create(&ctx, cemb, cemb),
        activation_create(cemb, ACT_SILU),
        linear_create(&ctx, cemb, cemb),
    };
    Sequential* embed = sequential_create(emb_mods, 4, max_batch);

    /* ---- UNet ---- */
    DDPMUNetConfig cfg = {
        .levels = levels,
        .channels = channels,
        .emb_dim = cemb,
        .heads = heads,
        .num_groups = num_groups,
        .cond_slot = cond_slot,
    };
    DDPMUNet* unet = ddpm_unet_create(&ctx, &cfg, C_img, C_img, H, W, max_batch);

    /* ---- optimizer ---- */
    ParamList params;
    param_list_init(&params);
    sequential_parameters(embed, &params);
    ddpm_unet_parameters(unet, &params);
    printf("total param entries: %d\n", params.count);
    Optimizer* opt = optimizer_adam(&params, lr, 0.9f, 0.999f, 1e-8f);

    /* ---- EMA ---- */
    float ema_decay = ini_float(&ini, "ema_decay", 0.9999f);
    EMA* ema = ema_create(&params, ema_decay);

    /* ---- loss ---- */
    Loss* loss = loss_create(max_batch);

    /* ---- device buffers ---- */
    float* d_clean = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_noisy = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_noise = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_pred = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_grad = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_grad_in = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_t_float = nn_device_alloc((size_t)max_batch);

    /* ========== DDPM Training (x0-prediction with min-SNR-gamma) ========== */
    int* h_t = (int*)malloc(max_batch * sizeof(int));
    float* h_t_float = (float*)malloc(max_batch * sizeof(float));
    float* h_w = (float*)malloc(max_batch * sizeof(float));

    printf("\n=== DDPM Training (%d steps, x0-prediction) ===\n", train_steps);
    for (int step = 1; step <= train_steps; ++step) {
        float cur_lr = lr_inv_sqrt(lr, step, lr_warmup, lr_ref);
        optimizer_set_lr(opt, cur_lr);
        for (int b = 0; b < max_batch; ++b) {
            int idx = rand() % n_train;
            NN_CUDA_CHECK(cudaMemcpy(d_clean + b * img_dim,
                h_images + idx * img_dim,
                img_dim * sizeof(float), cudaMemcpyHostToDevice));
        }
        for (int b = 0; b < max_batch; ++b) {
            h_t[b] = rand() % sched.T;
            h_t_float[b] = (float)h_t[b];
        }
        NN_CUDA_CHECK(cudaMemcpy(d_t_float, h_t_float, max_batch * sizeof(float),
            cudaMemcpyHostToDevice));
        nn_fill_normal(ctx.curand, d_noise, (size_t)max_batch * img_dim, 0.0f, 1.0f);

        for (int b = 0; b < max_batch; ++b) {
            launch_weighted_sum(d_noisy + b * img_dim,
                d_clean + b * img_dim,
                d_noise + b * img_dim,
                sched.sqrt_ab[h_t[b]], sched.sqrt_1mab[h_t[b]],
                img_dim, ctx.stream);
        }

        ctx_set_cond(&ctx, cond_slot, NULL);
        sequential_forward(embed, d_t_float, max_batch, &ctx);
        ctx_set_cond(&ctx, cond_slot, sequential_output(embed));

        ddpm_unet_forward(unet, d_noisy, d_pred, max_batch, &ctx);

        for (int b = 0; b < max_batch; ++b)
            h_w[b] = snr_weight[h_t[b]];
        float wmse = mse_weighted_forward_backward(loss, d_pred, d_clean,
            d_grad, max_batch, img_dim, h_w, &ctx);

        ctx_zero_cond_grad(&ctx, max_batch);
        ddpm_unet_backward(unet, d_noisy, d_pred, d_grad, d_grad_in, max_batch, &ctx);
        sequential_backward(embed, ctx.cond[cond_slot].grad, max_batch, &ctx);
        optimizer_step(opt, &params, &ctx);
        ema_update(ema, &params, &ctx);

        if (step == 1 || step % 500 == 0)
            printf("step %5d   wmse %.6f   lr %.2e\n", step, wmse, cur_lr);
    }

    /* ========== Diagnostic: model predictions at various noise levels ========== */
    {
        printf("\n=== Diagnostic: x0 prediction quality ===\n");
        NN_CUDA_CHECK(cudaMemcpy(d_clean, h_images, img_dim * sizeof(float),
            cudaMemcpyHostToDevice));
        for (int b = 1; b < max_batch; ++b)
            NN_CUDA_CHECK(cudaMemcpy(d_clean + b * img_dim, h_images,
                img_dim * sizeof(float), cudaMemcpyHostToDevice));

        nn_fill_normal(ctx.curand, d_noise, (size_t)max_batch * img_dim, 0.0f, 1.0f);

        int test_ts[] = { 0, 10, 50, 100, 200, 500, 800, 999 };
        int n_test = 8;
        float* h_pred = (float*)malloc(img_dim * sizeof(float));
        float* h_clean = (float*)malloc(img_dim * sizeof(float));
        NN_CUDA_CHECK(cudaMemcpy(h_clean, d_clean, img_dim * sizeof(float),
            cudaMemcpyDeviceToHost));

        for (int ti = 0; ti < n_test; ++ti) {
            int t = test_ts[ti];
            float c_noise = logf(sched.sqrt_1mab[t] / sched.sqrt_ab[t]) / 4.0f;
            for (int b = 0; b < max_batch; ++b)
                h_t_float[b] = c_noise;
            NN_CUDA_CHECK(cudaMemcpy(d_t_float, h_t_float, max_batch * sizeof(float),
                cudaMemcpyHostToDevice));

            for (int b = 0; b < max_batch; ++b)
                launch_weighted_sum(d_noisy + b * img_dim,
                    d_clean + b * img_dim,
                    d_noise + b * img_dim,
                    sched.sqrt_ab[t], sched.sqrt_1mab[t],
                    img_dim, ctx.stream);

            ctx_set_cond(&ctx, cond_slot, NULL);
            sequential_forward(embed, d_t_float, max_batch, &ctx);
            ctx_set_cond(&ctx, cond_slot, sequential_output(embed));
            ddpm_unet_forward(unet, d_noisy, d_pred, max_batch, &ctx);

            NN_CUDA_CHECK(cudaMemcpy(h_pred, d_pred, img_dim * sizeof(float),
                cudaMemcpyDeviceToHost));
            float mse = 0, pmin = h_pred[0], pmax = h_pred[0];
            for (int i = 0; i < img_dim; ++i) {
                float d = h_pred[i] - h_clean[i];
                mse += d * d;
                if (h_pred[i] < pmin)
                    pmin = h_pred[i];
                if (h_pred[i] > pmax)
                    pmax = h_pred[i];
            }
            mse /= img_dim;
            printf("  t=%3d  sqrt_ab=%.4f  pred range [%+.3f, %+.3f]  mse=%.6f\n",
                t, sched.sqrt_ab[t], pmin, pmax, mse);
        }
        free(h_pred);
        free(h_clean);
    }

    /* ========== DDIM Sampling ========== */
    mkdir("samples", 0755);
    float* h_samples = (float*)malloc((size_t)max_batch * img_dim * sizeof(float));
    DDIMChain* chain = ddim_chain_create(&sched, ddim_steps, img_dim, max_batch, cond_slot);

    /* sample with raw weights */
    printf("\n=== DDIM Sampling (raw weights, %d steps) ===\n", ddim_steps);
    ddim_chain_forward(chain, unet, embed, NULL, max_batch, &ctx);
    NN_CUDA_CHECK(cudaDeviceSynchronize());
    NN_CUDA_CHECK(cudaMemcpy(h_samples, ddim_chain_output(chain),
        (size_t)max_batch * img_dim * sizeof(float),
        cudaMemcpyDeviceToHost));
    float vmin = h_samples[0], vmax = h_samples[0];
    for (int i = 1; i < max_batch * img_dim; ++i) {
        if (h_samples[i] < vmin)
            vmin = h_samples[i];
        if (h_samples[i] > vmax)
            vmax = h_samples[i];
    }
    printf("  output range: [%.4f, %.4f]\n", vmin, vmax);
    save_pgm_grid("samples/ddim_raw.pgm", h_samples, max_batch, H, W, 4);

    /* sample with EMA weights */
    ema_swap(ema, &params, &ctx);
    printf("\n=== DDIM Sampling (EMA weights, %d steps) ===\n", ddim_steps);
    ddim_chain_forward(chain, unet, embed, NULL, max_batch, &ctx);
    NN_CUDA_CHECK(cudaDeviceSynchronize());
    NN_CUDA_CHECK(cudaMemcpy(h_samples, ddim_chain_output(chain),
        (size_t)max_batch * img_dim * sizeof(float),
        cudaMemcpyDeviceToHost));
    vmin = h_samples[0];
    vmax = h_samples[0];
    for (int i = 1; i < max_batch * img_dim; ++i) {
        if (h_samples[i] < vmin)
            vmin = h_samples[i];
        if (h_samples[i] > vmax)
            vmax = h_samples[i];
    }
    printf("  output range: [%.4f, %.4f]\n", vmin, vmax);
    save_pgm_grid("samples/ddim_ema.pgm", h_samples, max_batch, H, W, 4);
    ema_swap(ema, &params, &ctx);

    /* ---- cleanup ---- */
    free(h_t);
    free(h_t_float);
    free(h_w);
    free(h_samples);
    free(h_images);
    free(snr_weight);
    param_list_free(&params);
    cudaFree(d_clean);
    cudaFree(d_noisy);
    cudaFree(d_noise);
    cudaFree(d_pred);
    cudaFree(d_grad);
    cudaFree(d_grad_in);
    cudaFree(d_t_float);
    ddim_chain_free(chain);
    ddpm_schedule_free(&sched);
    ctx_free_cond(&ctx);
    loss_free(loss);
    optimizer_free(opt);
    ema_free(ema);
    ddpm_unet_free(unet);
    sequential_free(embed);
    curandDestroyGenerator(ctx.curand);
    cublasDestroy(ctx.cublas);
    return 0;
}
