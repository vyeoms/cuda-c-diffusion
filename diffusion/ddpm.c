#include "ddpm.h"
#include <math.h>
#include <stdlib.h>

/* ============================== schedule ============================== */

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

/* ============================== denoiser ============================== */

typedef struct {
    DDPMUNet* unet;
    Sequential* embed;
    int cond_slot, img_dim, max_batch;
    float* d_t; /* per-sample timestep, embedding input */
    float* d_grad_in; /* UNet input-grad sink (unused) */
} DDPMDenoiser;

static void ddpm_set_embedding(DDPMDenoiser* st, Context* ctx,
    const float* levels, int batch)
{
    NN_CUDA_CHECK(cudaMemcpyAsync(st->d_t, levels, (size_t)batch * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream));
    ctx_set_cond(ctx, st->cond_slot, NULL);
    sequential_forward(st->embed, st->d_t, batch, ctx);
    ctx_set_cond(ctx, st->cond_slot, sequential_output(st->embed));
}

static void ddpm_forward(Denoiser* self, Context* ctx, const float* x, float* out,
    int batch, const float* levels)
{
    DDPMDenoiser* st = (DDPMDenoiser*)self->state;
    NN_ASSERT(batch <= st->max_batch, "ddpm denoiser: batch exceeds max_batch");
    ddpm_set_embedding(st, ctx, levels, batch);
    ddpm_unet_forward(st->unet, x, out, batch, ctx);
}

static void ddpm_backward(Denoiser* self, Context* ctx, const float* x,
    const float* grad_out, int batch, const float* levels)
{
    (void)levels; /* x0-prediction: no per-sample coefficients in backward */
    DDPMDenoiser* st = (DDPMDenoiser*)self->state;
    ctx_zero_cond_grad(ctx, batch);
    ddpm_unet_backward(st->unet, x, NULL, grad_out, st->d_grad_in, batch, ctx);
    sequential_backward(st->embed, ctx->cond[st->cond_slot].grad, batch, ctx);
}

static void ddpm_parameters(Denoiser* self, ParamList* pl)
{
    DDPMDenoiser* st = (DDPMDenoiser*)self->state;
    sequential_parameters(st->embed, pl);
    ddpm_unet_parameters(st->unet, pl);
}

static void ddpm_denoiser_free(Denoiser* self)
{
    DDPMDenoiser* st = (DDPMDenoiser*)self->state;
    ddpm_unet_free(st->unet);
    sequential_free(st->embed);
    cudaFree(st->d_t);
    cudaFree(st->d_grad_in);
    free(st);
    free(self);
}

Denoiser* ddpm_denoiser_create(Context* ctx, const DDPMUNetConfig* unet_cfg,
    int C, int H, int W, int max_batch)
{
    Denoiser* self = (Denoiser*)malloc(sizeof(Denoiser));
    DDPMDenoiser* st = (DDPMDenoiser*)malloc(sizeof(DDPMDenoiser));

    DDPMUNetConfig cfg = *unet_cfg;
    cfg.cond_slot = ctx_add_cond(ctx, cfg.emb_dim, max_batch);

    Module* emb_mods[] = {
        sinusoidal_create(cfg.emb_dim, 10000.0f),
        linear_create(ctx, cfg.emb_dim, cfg.emb_dim),
        activation_create(cfg.emb_dim, ACT_SILU),
        linear_create(ctx, cfg.emb_dim, cfg.emb_dim),
    };
    st->embed = sequential_create(emb_mods, 4, max_batch);
    st->unet = ddpm_unet_create(ctx, &cfg, C, C, H, W, max_batch);

    st->cond_slot = cfg.cond_slot;
    st->img_dim = H * W * C;
    st->max_batch = max_batch;
    st->d_t = nn_device_alloc((size_t)max_batch);
    st->d_grad_in = nn_device_alloc((size_t)max_batch * st->img_dim);

    self->state = st;
    self->img_dim = st->img_dim;
    self->forward = ddpm_forward;
    self->backward = ddpm_backward;
    self->parameters = ddpm_parameters;
    self->free = ddpm_denoiser_free;
    return self;
}

/* ============================== training loss ============================== */

struct DDPMLoss {
    const DDPMSchedule* sched;
    int img_dim, max_batch, T;
    float* snr_weight; /* [T] min(snr, gamma)/gamma */
    Loss* mse;
    float* d_noise;
    float* d_noisy;
    float* d_pred;
    float* d_grad;
    int* h_t;
    float* h_t_float;
    float* h_w;
};

DDPMLoss* ddpm_loss_create(const DDPMSchedule* sched, float snr_gamma,
    int img_dim, int max_batch)
{
    DDPMLoss* l = (DDPMLoss*)malloc(sizeof(DDPMLoss));
    l->sched = sched;
    l->img_dim = img_dim;
    l->max_batch = max_batch;
    l->T = sched->T;
    l->mse = loss_create(max_batch);

    l->snr_weight = (float*)malloc((size_t)sched->T * sizeof(float));
    for (int t = 0; t < sched->T; ++t) {
        float snr = sched->alpha_bar[t] / (1.0f - sched->alpha_bar[t]);
        l->snr_weight[t] = fminf(snr, snr_gamma) / snr_gamma;
    }

    size_t n = (size_t)max_batch * img_dim;
    l->d_noise = nn_device_alloc(n);
    l->d_noisy = nn_device_alloc(n);
    l->d_pred = nn_device_alloc(n);
    l->d_grad = nn_device_alloc(n);
    l->h_t = (int*)malloc((size_t)max_batch * sizeof(int));
    l->h_t_float = (float*)malloc((size_t)max_batch * sizeof(float));
    l->h_w = (float*)malloc((size_t)max_batch * sizeof(float));
    return l;
}

float ddpm_loss_step(void* loss, Denoiser* den, const float* d_x0,
    int batch, Context* ctx)
{
    DDPMLoss* l = (DDPMLoss*)loss;
    int d = l->img_dim;
    const DDPMSchedule* s = l->sched;

    for (int b = 0; b < batch; ++b) {
        l->h_t[b] = rand() % l->T;
        l->h_t_float[b] = (float)l->h_t[b];
    }

    nn_fill_normal(ctx->curand, l->d_noise, (size_t)batch * d, 0.0f, 1.0f);
    for (int b = 0; b < batch; ++b)
        launch_weighted_sum(l->d_noisy + (size_t)b * d, d_x0 + (size_t)b * d,
            l->d_noise + (size_t)b * d,
            s->sqrt_ab[l->h_t[b]], s->sqrt_1mab[l->h_t[b]], d, ctx->stream);

    den->forward(den, ctx, l->d_noisy, l->d_pred, batch, l->h_t_float);

    for (int b = 0; b < batch; ++b)
        l->h_w[b] = l->snr_weight[l->h_t[b]];
    float mse = mse_weighted_forward_backward(l->mse, l->d_pred, d_x0,
        l->d_grad, batch, d, l->h_w, ctx);

    den->backward(den, ctx, l->d_noisy, l->d_grad, batch, l->h_t_float);
    return mse;
}

void ddpm_loss_free(DDPMLoss* l)
{
    loss_free(l->mse);
    free(l->snr_weight);
    cudaFree(l->d_noise);
    cudaFree(l->d_noisy);
    cudaFree(l->d_pred);
    cudaFree(l->d_grad);
    free(l->h_t);
    free(l->h_t_float);
    free(l->h_w);
    free(l);
}
