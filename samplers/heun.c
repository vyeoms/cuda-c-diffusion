#include "heun.h"
#include <math.h>
#include <stdlib.h>

void heun_sample(Denoiser* den, Context* ctx, float* x, int batch,
    float sigma_max, float sigma_min, int steps, int verbose)
{
    int n = batch * den->img_dim;
    float* d_denoised = nn_device_alloc((size_t)n);
    float* d_x_prev = nn_device_alloc((size_t)n);
    float* d_d = nn_device_alloc((size_t)n);

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

        denoiser_forward_uniform(den, ctx, x, d_denoised, batch, sig);

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

            denoiser_forward_uniform(den, ctx, x, d_denoised, batch, sig_next);
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
    cudaFree(d_denoised);
    cudaFree(d_x_prev);
    cudaFree(d_d);
}
