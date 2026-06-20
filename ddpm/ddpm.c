#include "ddpm.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ======================== schedule ======================== */

void ddpm_schedule_init(DDPMSchedule* s, int T, float beta_start, float beta_end)
{
    s->T = T;
    s->beta = (float*)malloc(T * sizeof(float));
    s->alpha = (float*)malloc(T * sizeof(float));
    s->alpha_bar = (float*)malloc(T * sizeof(float));
    s->sqrt_ab = (float*)malloc(T * sizeof(float));
    s->sqrt_1mab = (float*)malloc(T * sizeof(float));

    for (int t = 0; t < T; ++t) {
        s->beta[t] = beta_start + (float)t / (float)(T - 1) * (beta_end - beta_start);
        s->alpha[t] = 1.0f - s->beta[t];
    }
    s->alpha_bar[0] = 1.0f - beta_start;
    for (int t = 1; t < T; ++t)
        s->alpha_bar[t] = s->alpha_bar[t - 1] * s->alpha[t];
    for (int t = 0; t < T; ++t) {
        s->sqrt_ab[t] = sqrtf(s->alpha_bar[t]);
        s->sqrt_1mab[t] = sqrtf(1.0f - s->alpha_bar[t]);
    }
}

void ddpm_schedule_free(DDPMSchedule* s)
{
    free(s->beta);
    free(s->alpha);
    free(s->alpha_bar);
    free(s->sqrt_ab);
    free(s->sqrt_1mab);
}

/* ======================== DDIM chain (x0-prediction) ======================== */

struct DDIMChain {
    DDPMSchedule* sched;
    int S;
    int img_dim;
    int max_batch;
    int cond_slot;
    int* timesteps;

    float** chain_x;
    float** chain_x0;

    float* d_t;
};

DDIMChain* ddim_chain_create(DDPMSchedule* sched, int infer_steps,
    int img_dim, int max_batch, int cond_slot)
{
    DDIMChain* c = (DDIMChain*)calloc(1, sizeof(DDIMChain));
    c->sched = sched;
    c->S = infer_steps;
    c->img_dim = img_dim;
    c->max_batch = max_batch;
    c->cond_slot = cond_slot;

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

    c->d_t = nn_device_alloc((size_t)max_batch);
    return c;
}

static void chain_set_timestep(DDIMChain* c, int t, int batch, Sequential* embed,
    Context* ctx)
{
    float* h_t = (float*)malloc(batch * sizeof(float));
    for (int b = 0; b < batch; ++b)
        h_t[b] = (float)t;
    NN_CUDA_CHECK(cudaMemcpyAsync(c->d_t, h_t, batch * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream));
    free(h_t);
    ctx_set_cond(ctx, c->cond_slot, NULL);
    sequential_forward(embed, c->d_t, batch, ctx);
    ctx_set_cond(ctx, c->cond_slot, sequential_output(embed));
}

/* DDIM step (x0-prediction, deterministic eta=0):
     x0_pred = model(x_t, t)                           (model predicts clean image)
     eps     = (x_t - sqrt_ab[t] * x0_pred) / sqrt_1mab[t]  (derive noise)
     x_{t-1} = sqrt_ab[t-1] * x0_pred + sqrt_1mab[t-1] * eps

   Substituting eps:
     x_{t-1} = coeff_x0 * x0_pred + coeff_xt * x_t
   where coeff_x0 = sqrt_ab_prev - sqrt_1mab_prev * sqrt_ab_t / sqrt_1mab_t
         coeff_xt = sqrt_1mab_prev / sqrt_1mab_t                              */
void ddim_chain_forward(DDIMChain* c, DDPMUNet* unet, Sequential* embed,
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

        chain_set_timestep(c, t, batch, embed, ctx);
        ddpm_unet_forward(unet, c->chain_x[i], c->chain_x0[i], batch, ctx);

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
    cudaFree(c->d_t);
    free(c->chain_x);
    free(c->chain_x0);
    free(c->timesteps);
    free(c);
}
