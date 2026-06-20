#include "ini.h"
#include "mnist_io.h"
#include "nn.h"
#include "pgm_io.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#define TWO_PI 6.2831853f

typedef struct {
    UNet* unet;
    Sequential* embed;
    float* d_sigma;
    float* d_scaled;
    int img_dim;
    float sigma_data;
    int cond_slot;
} Denoiser;

static void edm_coeffs(float sigma, float sd, float* cin, float* cskip, float* cout)
{
    float s2 = sigma * sigma, sd2 = sd * sd;
    *cin = 1.0f / sqrtf(s2 + sd2);
    *cskip = sd2 / (s2 + sd2);
    *cout = sigma * sd / sqrtf(s2 + sd2);
}

static void denoise(const Denoiser* den, Context* ctx,
    const float* x, float* out, int batch, float sigma)
{
    float ci, cs, co;
    edm_coeffs(sigma, den->sigma_data, &ci, &cs, &co);
    float cnoise = logf(sigma) / 4.0f;
    int n = batch * den->img_dim;

    launch_weighted_sum(den->d_scaled, x, x, ci, 0.0f, n, ctx->stream);

    float* h_cn = (float*)malloc(batch * sizeof(float));
    for (int b = 0; b < batch; ++b)
        h_cn[b] = cnoise;
    NN_CUDA_CHECK(cudaMemcpyAsync(den->d_sigma, h_cn, batch * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream));
    free(h_cn);
    ctx_set_cond(ctx, den->cond_slot, NULL);
    sequential_forward(den->embed, den->d_sigma, batch, ctx);
    ctx_set_cond(ctx, den->cond_slot, sequential_output(den->embed));

    unet_forward(den->unet, den->d_scaled, out, batch, ctx);
    launch_weighted_sum(out, x, out, cs, co, n, ctx->stream);
}

static void heun_sample(const Denoiser* den, Context* ctx,
    float* x, int batch,
    float sigma_max, float sigma_min, int steps,
    float* d_denoised, float* d_x_prev, float* d_d,
    int verbose)
{
    int n = batch * den->img_dim;
    nn_fill_normal(ctx->curand, x, (size_t)n, 0.0f, sigma_max);

    float* h_diag = verbose ? (float*)malloc(n * sizeof(float)) : NULL;

    for (int i = 0; i < steps; ++i) {
        float t0 = (float)i / (float)steps;
        float t1 = (float)(i + 1) / (float)steps;
        float sig = sigma_max * powf(sigma_min / sigma_max, t0);
        float sig_next = (i + 1 < steps)
            ? sigma_max * powf(sigma_min / sigma_max, t1)
            : 0.0f;
        float dt = sig_next - sig;

        denoise(den, ctx, x, d_denoised, batch, sig);

        if (verbose && (i < 5 || i == steps - 1 || i == steps / 2)) {
            NN_CUDA_CHECK(cudaDeviceSynchronize());
            float xm = 0, dm = 0;
            NN_CUDA_CHECK(cudaMemcpy(h_diag, x, n * sizeof(float),
                cudaMemcpyDeviceToHost));
            for (int j = 0; j < n; ++j)
                xm += h_diag[j];
            NN_CUDA_CHECK(cudaMemcpy(h_diag, d_denoised, n * sizeof(float),
                cudaMemcpyDeviceToHost));
            for (int j = 0; j < n; ++j)
                dm += h_diag[j];
            printf("  step %2d  sig=%.4f  x_mean=%.4f  D_mean=%.4f\n",
                i, sig, xm / n, dm / n);
        }

        launch_weighted_sum(d_d, x, d_denoised, 1.0f / sig, -1.0f / sig,
            n, ctx->stream);

        if (sig_next > 0.0f) {
            NN_CUDA_CHECK(cudaMemcpyAsync(d_x_prev, x, n * sizeof(float),
                cudaMemcpyDeviceToDevice, ctx->stream));
            launch_weighted_sum(x, x, d_d, 1.0f, dt, n, ctx->stream);

            denoise(den, ctx, x, d_denoised, batch, sig_next);
            launch_weighted_sum(d_denoised, x, d_denoised,
                1.0f / sig_next, -1.0f / sig_next,
                n, ctx->stream);
            launch_weighted_sum(d_d, d_d, d_denoised, 0.5f, 0.5f, n, ctx->stream);
            launch_weighted_sum(x, d_x_prev, d_d, 1.0f, dt, n, ctx->stream);
        } else {
            NN_CUDA_CHECK(cudaMemcpyAsync(x, d_denoised, n * sizeof(float),
                cudaMemcpyDeviceToDevice, ctx->stream));
        }
    }
    free(h_diag);
}

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
    int mp_mode = ini_int(&ini, "mp_mode", 1);
    int heads = ini_int(&ini, "heads", 4);

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

    /* ---- embedding: mp_fourier → mp_linear → mp_silu ---- */
    Module* emb_mods[] = {
        mp_fourier_create(&ctx, cemb, 1.0f),
        mp_linear_create(&ctx, cemb, cemb, 1.0f),
        activation_create(cemb, ACT_MP_SILU),
    };
    Sequential* embed = sequential_create(emb_mods, 3, max_batch);

    /* ---- UNet ---- */
    UNetConfig cfg = {
        .levels = levels,
        .channels = channels,
        .emb_dim = cemb,
        .heads = heads,
        .t = t_mp,
        .mp_mode = mp_mode,
        .cond_slot = cond_slot,
    };
    UNet* unet = unet_create(&ctx, &cfg, C_img, C_img, H, W, max_batch);

    /* ---- optimizer ---- */
    ParamList params;
    param_list_init(&params);
    sequential_parameters(embed, &params);
    unet_parameters(unet, &params);
    printf("total param entries: %d\n", params.count);
    Optimizer* opt = optimizer_adam(&params, lr, 0.9f, 0.999f, 1e-8f);

    /* ---- EMA ---- */
    EMA* ema = ema_create(&params, ema_decay);

    /* ---- loss ---- */
    Loss* loss = loss_create(max_batch);

    /* ---- device buffers ---- */
    float* d_clean = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_noisy = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_noise = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_sigma = nn_device_alloc((size_t)max_batch);
    float* d_pred = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_grad = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_grad_in = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_scaled = nn_device_alloc((size_t)max_batch * img_dim);
    float* d_target = nn_device_alloc((size_t)max_batch * img_dim);

    if (vis_batch > max_batch)
        vis_batch = max_batch;
    Denoiser den = { unet, embed, d_sigma, d_scaled, img_dim, sigma_data, cond_slot };

    /* ========== Training ========== */
    float* h_sigma = (float*)malloc(max_batch * sizeof(float));
    float* h_cnoise = (float*)malloc(max_batch * sizeof(float));
    float* h_cin = (float*)malloc(max_batch * sizeof(float));
    float* h_cskip = (float*)malloc(max_batch * sizeof(float));
    float* h_cout = (float*)malloc(max_batch * sizeof(float));

    printf("\n=== EDM Training (%d steps, batch=%d) ===\n", train_steps, max_batch);
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
            float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
            float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
            float z = sqrtf(-2.0f * logf(u1)) * cosf(TWO_PI * u2);
            float ln_sig = p_mean + p_std * z;
            h_sigma[b] = fminf(fmaxf(expf(ln_sig), sigma_min), sigma_max);
            h_cnoise[b] = logf(h_sigma[b]) / 4.0f;
            edm_coeffs(h_sigma[b], sigma_data, &h_cin[b], &h_cskip[b], &h_cout[b]);
        }

        nn_fill_normal(ctx.curand, d_noise, (size_t)max_batch * img_dim, 0.0f, 1.0f);

        for (int b = 0; b < max_batch; ++b)
            launch_weighted_sum(d_noisy + b * img_dim,
                d_clean + b * img_dim,
                d_noise + b * img_dim,
                1.0f, h_sigma[b], img_dim, ctx.stream);

        for (int b = 0; b < max_batch; ++b)
            launch_weighted_sum(d_scaled + b * img_dim,
                d_noisy + b * img_dim,
                d_noisy + b * img_dim,
                h_cin[b], 0.0f, img_dim, ctx.stream);

        NN_CUDA_CHECK(cudaMemcpy(d_sigma, h_cnoise, max_batch * sizeof(float),
            cudaMemcpyHostToDevice));
        ctx_set_cond(&ctx, cond_slot, NULL);
        sequential_forward(embed, d_sigma, max_batch, &ctx);
        ctx_set_cond(&ctx, cond_slot, sequential_output(embed));

        unet_forward(unet, d_scaled, d_pred, max_batch, &ctx);

        for (int b = 0; b < max_batch; ++b)
            launch_weighted_sum(d_target + b * img_dim,
                d_clean + b * img_dim,
                d_noisy + b * img_dim,
                1.0f / h_cout[b], -h_cskip[b] / h_cout[b],
                img_dim, ctx.stream);

        float mse = mse_forward_backward(loss, d_pred, d_target,
            d_grad, max_batch, img_dim, &ctx);

        ctx_zero_cond_grad(&ctx, max_batch);
        unet_backward(unet, d_scaled, d_pred, d_grad, d_grad_in, max_batch, &ctx);
        sequential_backward(embed, ctx.cond[cond_slot].grad, max_batch, &ctx);
        optimizer_step(opt, &params, &ctx);
        ema_update(ema, &params, &ctx);

        if (step == 1 || step % 1000 == 0)
            printf("step %5d   mse %.6f   lr %.2e\n", step, mse, cur_lr);
    }

    /* ========== Sampling (use EMA weights) ========== */
    ema_swap(ema, &params, &ctx);

    printf("\n=== Heun ODE Sampling (%d steps) ===\n", sample_steps);
    mkdir("samples", 0755);

    heun_sample(&den, &ctx, d_noisy, vis_batch,
        sigma_max, sigma_min, sample_steps,
        d_pred, d_grad, d_noise, 1);
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

    denoise(&den, &ctx, d_noisy, d_pred, vis_batch, demo_sigma);
    NN_CUDA_CHECK(cudaDeviceSynchronize());

    save_vis("samples/originals.pgm", d_clean, h_buf,
        vis_batch, H, W, 4, scale, data_mean);
    save_vis("samples/noisy.pgm", d_noisy, h_buf,
        vis_batch, H, W, 4, scale, data_mean);
    save_vis("samples/denoised.pgm", d_pred, h_buf,
        vis_batch, H, W, 4, scale, data_mean);

    ema_swap(ema, &params, &ctx);

    /* ---- cleanup ---- */
    free(h_sigma);
    free(h_cnoise);
    free(h_cin);
    free(h_cskip);
    free(h_cout);
    free(h_buf);
    free(h_images);
    param_list_free(&params);
    cudaFree(d_clean);
    cudaFree(d_noisy);
    cudaFree(d_noise);
    cudaFree(d_sigma);
    cudaFree(d_pred);
    cudaFree(d_grad);
    cudaFree(d_grad_in);
    cudaFree(d_scaled);
    cudaFree(d_target);
    ctx_free_cond(&ctx);
    loss_free(loss);
    optimizer_free(opt);
    ema_free(ema);
    unet_free(unet);
    sequential_free(embed);
    curandDestroyGenerator(ctx.curand);
    cublasDestroy(ctx.cublas);
    return 0;
}
