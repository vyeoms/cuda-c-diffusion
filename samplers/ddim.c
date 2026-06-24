#include "ddim.h"
#include <stdlib.h>

struct DDIMChain {
    const DDPMSchedule* sched;
    int S;
    int img_dim;
    int max_batch;
    int* timesteps;

    float** chain_x; /* S+1 */
    float** chain_x0; /* S   */
};

DDIMChain* ddim_chain_create(const DDPMSchedule* sched, int infer_steps,
    int img_dim, int max_batch)
{
    DDIMChain* c = (DDIMChain*)calloc(1, sizeof(DDIMChain));
    c->sched = sched;
    c->S = infer_steps;
    c->img_dim = img_dim;
    c->max_batch = max_batch;

    c->timesteps = (int*)malloc((infer_steps + 1) * sizeof(int));
    for (int i = 0; i <= infer_steps; ++i)
        c->timesteps[i] = (int)((float)(infer_steps - i) / (float)infer_steps
                * (float)(sched->T - 1)
            + 0.5f);

    size_t buf = (size_t)max_batch * img_dim;
    c->chain_x = (float**)malloc((infer_steps + 1) * sizeof(float*));
    c->chain_x0 = (float**)malloc(infer_steps * sizeof(float*));
    for (int i = 0; i <= infer_steps; ++i)
        c->chain_x[i] = nn_device_alloc(buf);
    for (int i = 0; i < infer_steps; ++i)
        c->chain_x0[i] = nn_device_alloc(buf);
    return c;
}

/* DDIM step (x0-prediction, deterministic eta=0):
     x0_pred = D(x_t, t)                               (denoiser yields clean x0)
     eps     = (x_t - sqrt_ab[t] * x0_pred) / sqrt_1mab[t]
     x_{t-1} = sqrt_ab[t-1] * x0_pred + sqrt_1mab[t-1] * eps
   Substituting eps:
     x_{t-1} = coeff_x0 * x0_pred + coeff_xt * x_t
   where coeff_x0 = sqrt_ab_prev - sqrt_1mab_prev * sqrt_ab_t / sqrt_1mab_t
         coeff_xt = sqrt_1mab_prev / sqrt_1mab_t                              */
void ddim_chain_forward(DDIMChain* c, Denoiser* den,
    const float* x_T, int batch, Context* ctx)
{
    int n = batch * c->img_dim;

    if (x_T)
        NN_CUDA_CHECK(cudaMemcpyAsync(c->chain_x[0], x_T, n * sizeof(float),
            cudaMemcpyDeviceToDevice, ctx->stream));
    else
        nn_fill_normal(ctx->curand, c->chain_x[0], (size_t)n, 0.0f, 1.0f);

    for (int i = 0; i < c->S; ++i) {
        int t = c->timesteps[i];
        int t_next = c->timesteps[i + 1];

        denoiser_forward_uniform(den, ctx, c->chain_x[i], c->chain_x0[i],
            batch, (float)t);

        if (t_next == 0) {
            NN_CUDA_CHECK(cudaMemcpyAsync(c->chain_x[i + 1], c->chain_x0[i],
                n * sizeof(float), cudaMemcpyDeviceToDevice, ctx->stream));
        } else {
            float sab_t = c->sched->sqrt_ab[t];
            float s1mab_t = c->sched->sqrt_1mab[t];
            float sab_next = c->sched->sqrt_ab[t_next];
            float s1mab_next = c->sched->sqrt_1mab[t_next];
            float coeff_x0 = sab_next - s1mab_next * sab_t / s1mab_t;
            float coeff_xt = s1mab_next / s1mab_t;
            launch_weighted_sum(c->chain_x[i + 1], c->chain_x0[i], c->chain_x[i],
                coeff_x0, coeff_xt, n, ctx->stream);
        }
    }
}

float* ddim_chain_output(DDIMChain* c)
{
    return c->chain_x[c->S];
}

void ddim_chain_free(DDIMChain* c)
{
    for (int i = 0; i <= c->S; ++i)
        cudaFree(c->chain_x[i]);
    for (int i = 0; i < c->S; ++i)
        cudaFree(c->chain_x0[i]);
    free(c->chain_x);
    free(c->chain_x0);
    free(c->timesteps);
    free(c);
}
