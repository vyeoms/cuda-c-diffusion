#include "ddim.h" /* DDIM sampler (samplers/) */
#include "ddpm.h" /* DDPM schedule + denoiser + loss (diffusion/) */
#include "ini.h"
#include "mnist_io.h"
#include "nn.h"
#include "pgm_io.h"
#include "train.h" /* shared training loop (diffusion/) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

static float range_min(const float* h, int n)
{
    float m = h[0];
    for (int i = 1; i < n; ++i)
        if (h[i] < m)
            m = h[i];
    return m;
}
static float range_max(const float* h, int n)
{
    float m = h[0];
    for (int i = 1; i < n; ++i)
        if (h[i] > m)
            m = h[i];
    return m;
}

/* x0-prediction quality at a sweep of noise levels, routed through the denoiser
   (so it uses the same timestep conditioning as training and sampling). */
static void diagnose_x0(Denoiser* den, const DDPMSchedule* sched,
    const float* h_image, int img_dim, int max_batch, Context* ctx,
    float* d_clean, float* d_noisy, float* d_noise, float* d_pred)
{
    printf("\n=== Diagnostic: x0 prediction quality ===\n");
    for (int b = 0; b < max_batch; ++b)
        NN_CUDA_CHECK(cudaMemcpy(d_clean + b * img_dim, h_image,
            img_dim * sizeof(float), cudaMemcpyHostToDevice));
    nn_fill_normal(ctx->curand, d_noise, (size_t)max_batch * img_dim, 0.0f, 1.0f);

    int test_ts[] = { 0, 10, 50, 100, 200, 500, 800, 999 };
    float* h_pred = (float*)malloc(img_dim * sizeof(float));

    for (int ti = 0; ti < 8; ++ti) {
        int t = test_ts[ti];
        for (int b = 0; b < max_batch; ++b)
            launch_weighted_sum(d_noisy + b * img_dim, d_clean + b * img_dim,
                d_noise + b * img_dim,
                sched->sqrt_ab[t], sched->sqrt_1mab[t], img_dim, ctx->stream);

        denoiser_forward_uniform(den, ctx, d_noisy, d_pred, max_batch, (float)t);
        NN_CUDA_CHECK(cudaMemcpy(h_pred, d_pred, img_dim * sizeof(float),
            cudaMemcpyDeviceToHost));

        float mse = 0;
        for (int i = 0; i < img_dim; ++i) {
            float dlt = h_pred[i] - h_image[i];
            mse += dlt * dlt;
        }
        mse /= img_dim;
        printf("  t=%3d  sqrt_ab=%.4f  pred range [%+.3f, %+.3f]  mse=%.6f\n",
            t, sched->sqrt_ab[t], range_min(h_pred, img_dim),
            range_max(h_pred, img_dim), mse);
    }
    free(h_pred);
}

static void sample_and_save(const char* tag, const char* path, DDIMChain* chain,
    Denoiser* den, int max_batch, int img_dim, int H, int W,
    float* h_samples, Context* ctx)
{
    printf("\n=== DDIM Sampling (%s) ===\n", tag);
    ddim_chain_forward(chain, den, NULL, max_batch, ctx);
    NN_CUDA_CHECK(cudaDeviceSynchronize());
    NN_CUDA_CHECK(cudaMemcpy(h_samples, ddim_chain_output(chain),
        (size_t)max_batch * img_dim * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  output range: [%.4f, %.4f]\n",
        range_min(h_samples, max_batch * img_dim),
        range_max(h_samples, max_batch * img_dim));
    save_pgm_grid(path, h_samples, max_batch, H, W, 4);
}

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
    float ema_decay = ini_float(&ini, "ema_decay", 0.9999f);

    int seed = ini_int(&ini, "seed", 410293);
    int tf32 = ini_int(&ini, "tf32", 1);
    int save_every = ini_int(&ini, "save_every", 0);
    const char* ckpt_dir = ini_str(&ini, "ckpt_dir", "checkpoints");
    const char* resume = ini_str(&ini, "resume", "");

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

    /* ---- CUDA context (TF32 is faster but not bit-reproducible; see README) ---- */
    srand((unsigned)seed);
    Context ctx;
    NN_CUBLAS_CHECK(cublasCreate(&ctx.cublas));
    if (tf32)
        NN_CUBLAS_CHECK(cublasSetMathMode(ctx.cublas, CUBLAS_TF32_TENSOR_OP_MATH));
    ctx.stream = 0;
    NN_CURAND_CHECK(curandCreateGenerator(&ctx.curand, CURAND_RNG_PSEUDO_DEFAULT));
    NN_CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx.curand, (unsigned long long)seed));
    NN_CURAND_CHECK(curandSetStream(ctx.curand, ctx.stream));
    ctx.n_cond = 0;

    /* ---- DDPM schedule + denoiser (x0-prediction UNet + timestep embedding) ---- */
    DDPMSchedule sched;
    ddpm_schedule_init(&sched, sched_T, beta_start, beta_end);

    DDPMUNetConfig cfg = {
        .levels = levels,
        .channels = channels,
        .emb_dim = cemb,
        .heads = heads,
        .num_groups = num_groups,
        .cond_slot = -1, /* assigned by ddpm_denoiser_create */
    };
    Denoiser* den = ddpm_denoiser_create(&ctx, &cfg, C_img, H, W, max_batch);

    /* ---- optimizer / EMA over the denoiser's parameters ---- */
    ParamList params;
    param_list_init(&params);
    den->parameters(den, &params);
    printf("total param entries: %d\n", params.count);
    Optimizer* opt = optimizer_adam(&params, lr, 0.9f, 0.999f, 1e-8f);
    EMA* ema = ema_create(&params, ema_decay);

    /* ---- DDPM training loss (x0-prediction + min-SNR-gamma) ---- */
    DDPMLoss* dloss = ddpm_loss_create(&sched, snr_gamma, img_dim, max_batch);

    /* ---- buffers owned by main (batch staging + diagnostics) ---- */
    float* d_clean = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_noisy = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_noise = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_pred = nn_device_alloc((size_t)max_batch * img_dim);

    /* ========== Training ========== */
    TrainConfig tcfg = {
        .train_steps = train_steps,
        .lr = lr,
        .lr_warmup = lr_warmup,
        .lr_ref = lr_ref,
        .log_every = 500,
        .loss_label = "wmse",
        .save_every = save_every,
        .ckpt_dir = ckpt_dir,
        .resume_path = resume,
    };
    train_diffusion(den, &params, opt, ema, dloss, ddpm_loss_step,
        h_images, n_train, img_dim, max_batch, tcfg, &ctx);

    /* ========== Diagnostics + sampling ========== */
    diagnose_x0(den, &sched, h_images, img_dim, max_batch, &ctx,
        d_clean, d_noisy, d_noise, d_pred);

    mkdir("samples", 0755);
    float* h_samples = (float*)malloc((size_t)max_batch * img_dim * sizeof(float));
    DDIMChain* chain = ddim_chain_create(&sched, ddim_steps, img_dim, max_batch);

    sample_and_save("raw weights", "samples/ddim_raw.pgm", chain, den,
        max_batch, img_dim, H, W, h_samples, &ctx);

    ema_swap(ema, &params, &ctx);
    sample_and_save("EMA weights", "samples/ddim_ema.pgm", chain, den,
        max_batch, img_dim, H, W, h_samples, &ctx);
    ema_swap(ema, &params, &ctx);

    /* ---- cleanup ---- */
    free(h_samples);
    free(h_images);
    param_list_free(&params);
    cudaFree(d_clean);
    cudaFree(d_noisy);
    cudaFree(d_noise);
    cudaFree(d_pred);
    ddim_chain_free(chain);
    ddpm_loss_free(dloss);
    ddpm_schedule_free(&sched);
    optimizer_free(opt);
    ema_free(ema);
    den->free(den);
    ctx_free_cond(&ctx);
    curandDestroyGenerator(ctx.curand);
    cublasDestroy(ctx.cublas);
    return 0;
}
