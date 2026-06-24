#include "edm.h" /* EDM denoiser + loss (diffusion/) */
#include "heun.h" /* Heun ODE sampler (samplers/) */
#include "ini.h"
#include "mnist_io.h"
#include "nn.h"
#include "pgm_io.h"
#include "train.h" /* shared training loop (diffusion/) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

static void unnormalize(float* buf, int n, float scale, float data_mean)
{
    for (int i = 0; i < n; ++i)
        buf[i] = buf[i] / scale + data_mean;
}

static void save_vis(const char* path, const float* d_src, float* h_buf,
    int count, int H, int W, int cols,
    float scale, float data_mean)
{
    int n = count * H * W;
    NN_CUDA_CHECK(cudaMemcpy(h_buf, d_src, n * sizeof(float),
        cudaMemcpyDeviceToHost));
    unnormalize(h_buf, n, scale, data_mean);
    save_pgm_grid(path, h_buf, count, H, W, cols);
}

int main(int argc, char** argv)
{
    const char* ini_path = (argc > 1) ? argv[1] : "edm/edm.ini";
    Ini ini;
    if (ini_load(&ini, ini_path) != 0) {
        fprintf(stderr, "failed to load config: %s\n", ini_path);
        return 1;
    }
    printf("config: %s\n", ini_path);

    int max_batch = ini_int(&ini, "batch_size", 64);
    int vis_batch = ini_int(&ini, "vis_batch", 16);
    int cemb = ini_int(&ini, "emb_dim", 64);
    int train_steps = ini_int(&ini, "train_steps", 20000);
    int sample_steps = ini_int(&ini, "sample_steps", 50);
    float lr = ini_float(&ini, "lr", 1e-3f);
    int lr_warmup = ini_int(&ini, "lr_warmup", 500);
    int lr_ref = ini_int(&ini, "lr_ref", 5000);
    float demo_sigma = ini_float(&ini, "demo_sigma", 0.5f);
    float ema_decay = ini_float(&ini, "ema_decay", 0.9999f);

    float sigma_data = ini_float(&ini, "sigma_data", 0.5f);
    float p_mean = ini_float(&ini, "p_mean", -1.2f);
    float p_std = ini_float(&ini, "p_std", 1.2f);
    float sigma_min = ini_float(&ini, "sigma_min", 0.002f);
    float sigma_max = ini_float(&ini, "sigma_max", 80.0f);

    int H = ini_int(&ini, "H", 28);
    int W = ini_int(&ini, "W", 28);
    int C_img = ini_int(&ini, "C", 1);
    float t_mp = ini_float(&ini, "t", 0.3f);
    int heads = ini_int(&ini, "heads", 4);

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

    /* ---- preprocess: zero-mean, std = sigma_data ---- */
    double sum = 0, sum2 = 0;
    size_t total = (size_t)n_train * img_dim;
    for (size_t i = 0; i < total; ++i)
        sum += h_images[i];
    float data_mean = (float)(sum / total);
    for (size_t i = 0; i < total; ++i)
        sum2 += (h_images[i] - data_mean) * (h_images[i] - data_mean);
    float data_std = (float)sqrt(sum2 / total);
    float scale = sigma_data / data_std;
    for (size_t i = 0; i < total; ++i)
        h_images[i] = (h_images[i] - data_mean) * scale;
    printf("preprocessing: mean=%.4f std=%.4f -> zero-mean, std=%.1f\n",
        data_mean, data_std, sigma_data);

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

    /* ---- EDM denoiser (UNet + Fourier time embedding + preconditioning) ---- */
    UNetConfig cfg = {
        .levels = levels,
        .channels = channels,
        .emb_dim = cemb,
        .heads = heads,
        .t = t_mp,
        .cond_slot = -1, /* assigned by edm_denoiser_create */
    };
    Denoiser* den = edm_denoiser_create(&ctx, &cfg, C_img, H, W, max_batch, sigma_data);

    /* ---- optimizer / EMA over the denoiser's parameters ---- */
    ParamList params;
    param_list_init(&params);
    den->parameters(den, &params);
    printf("total param entries: %d\n", params.count);
    Optimizer* opt = optimizer_adam(&params, lr, 0.9f, 0.999f, 1e-8f);
    EMA* ema = ema_create(&params, ema_decay);

    /* ---- EDM training loss ---- */
    EDMLossCfg lcfg = { p_mean, p_std, sigma_min, sigma_max, sigma_data };
    EDMLoss* eloss = edm_loss_create(img_dim, max_batch, lcfg);

    /* ---- buffers owned by main (sampling + demo) ---- */
    float* d_clean = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_noisy = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_noise = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_pred = nn_device_alloc((size_t)max_batch * img_dim);

    if (vis_batch > max_batch)
        vis_batch = max_batch;

    /* ========== Training ========== */
    TrainConfig tcfg = {
        .train_steps = train_steps,
        .lr = lr,
        .lr_warmup = lr_warmup,
        .lr_ref = lr_ref,
        .log_every = 1000,
        .loss_label = "mse",
        .save_every = save_every,
        .ckpt_dir = ckpt_dir,
        .resume_path = resume,
    };
    train_diffusion(den, &params, opt, ema, eloss, edm_loss_step,
        h_images, n_train, img_dim, max_batch, tcfg, &ctx);

    /* ========== Sampling (use EMA weights) ========== */
    ema_swap(ema, &params, &ctx);

    printf("\n=== Heun ODE Sampling (%d steps) ===\n", sample_steps);
    mkdir("samples", 0755);

    heun_sample(den, &ctx, d_noisy, vis_batch,
        sigma_max, sigma_min, sample_steps, 1);
    NN_CUDA_CHECK(cudaDeviceSynchronize());

    float* h_buf = (float*)malloc((size_t)vis_batch * img_dim * sizeof(float));
    save_vis("samples/heun_samples.pgm", d_noisy, h_buf,
        vis_batch, H, W, 4, scale, data_mean);

    /* ========== Denoising Demo ========== */
    printf("\n=== Denoising Examples (sigma=%.2f) ===\n", demo_sigma);
    for (int b = 0; b < vis_batch; ++b)
        NN_CUDA_CHECK(cudaMemcpy(d_clean + b * img_dim,
            h_images + b * img_dim,
            img_dim * sizeof(float), cudaMemcpyHostToDevice));

    nn_fill_normal(ctx.curand, d_noise, (size_t)vis_batch * img_dim, 0.0f, 1.0f);
    for (int b = 0; b < vis_batch; ++b)
        launch_weighted_sum(d_noisy + b * img_dim,
            d_clean + b * img_dim,
            d_noise + b * img_dim,
            1.0f, demo_sigma, img_dim, ctx.stream);

    denoiser_forward_uniform(den, &ctx, d_noisy, d_pred, vis_batch, demo_sigma);
    NN_CUDA_CHECK(cudaDeviceSynchronize());

    save_vis("samples/originals.pgm", d_clean, h_buf,
        vis_batch, H, W, 4, scale, data_mean);
    save_vis("samples/noisy.pgm", d_noisy, h_buf,
        vis_batch, H, W, 4, scale, data_mean);
    save_vis("samples/denoised.pgm", d_pred, h_buf,
        vis_batch, H, W, 4, scale, data_mean);

    ema_swap(ema, &params, &ctx);

    /* ---- cleanup ---- */
    free(h_buf);
    free(h_images);
    param_list_free(&params);
    cudaFree(d_clean);
    cudaFree(d_noisy);
    cudaFree(d_noise);
    cudaFree(d_pred);
    edm_loss_free(eloss);
    optimizer_free(opt);
    ema_free(ema);
    den->free(den);
    ctx_free_cond(&ctx);
    curandDestroyGenerator(ctx.curand);
    cublasDestroy(ctx.cublas);
    return 0;
}
