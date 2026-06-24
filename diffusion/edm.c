#include "edm.h"
#include <math.h>
#include <stdlib.h>

#define TWO_PI 6.2831853f

static void edm_coeffs(float sigma, float sd, float* cin, float* cskip, float* cout)
{
    float s2 = sigma * sigma, sd2 = sd * sd;
    *cin = 1.0f / sqrtf(s2 + sd2);
    *cskip = sd2 / (s2 + sd2);
    *cout = sigma * sd / sqrtf(s2 + sd2);
}

/* ============================== denoiser ============================== */

typedef struct {
    UNet* unet;
    Sequential* embed;
    int cond_slot;
    int img_dim, max_batch;
    float sigma_data;

    float* d_scaled; /* c_in * x, fed to the UNet (kept for backward) */
    float* d_F; /* UNet output F (kept for backward) */
    float* d_cnoise; /* per-sample c_noise, embedding input */
    float* d_gF; /* dL/dF */
    float* d_grad_in; /* UNet input-grad sink (unused) */

    float *h_cin, *h_cskip, *h_cout, *h_cnoise;
} EDMDenoiser;

static void edm_set_embedding(EDMDenoiser* st, Context* ctx, int batch)
{
    NN_CUDA_CHECK(cudaMemcpyAsync(st->d_cnoise, st->h_cnoise,
        (size_t)batch * sizeof(float), cudaMemcpyHostToDevice, ctx->stream));
    ctx_set_cond(ctx, st->cond_slot, NULL);
    sequential_forward(st->embed, st->d_cnoise, batch, ctx);
    ctx_set_cond(ctx, st->cond_slot, sequential_output(st->embed));
}

static void edm_forward(Denoiser* self, Context* ctx, const float* x, float* out,
    int batch, const float* levels)
{
    EDMDenoiser* st = (EDMDenoiser*)self->state;
    NN_ASSERT(batch <= st->max_batch, "edm denoiser: batch exceeds max_batch");
    int d = st->img_dim;

    for (int b = 0; b < batch; ++b) {
        edm_coeffs(levels[b], st->sigma_data,
            &st->h_cin[b], &st->h_cskip[b], &st->h_cout[b]);
        st->h_cnoise[b] = logf(levels[b]) / 4.0f;
    }
    edm_set_embedding(st, ctx, batch);

    for (int b = 0; b < batch; ++b)
        launch_weighted_sum(st->d_scaled + (size_t)b * d, x + (size_t)b * d,
            x + (size_t)b * d, st->h_cin[b], 0.0f, d, ctx->stream);

    unet_forward(st->unet, st->d_scaled, st->d_F, batch, ctx);

    for (int b = 0; b < batch; ++b)
        launch_weighted_sum(out + (size_t)b * d, x + (size_t)b * d,
            st->d_F + (size_t)b * d, st->h_cskip[b], st->h_cout[b], d, ctx->stream);
}

static void edm_backward(Denoiser* self, Context* ctx, const float* x,
    const float* grad_out, int batch, const float* levels)
{
    (void)x;
    EDMDenoiser* st = (EDMDenoiser*)self->state;
    int d = st->img_dim;

    /* D = c_skip*x + c_out*F  =>  dL/dF = c_out * dL/dD */
    for (int b = 0; b < batch; ++b) {
        float cin, cskip, cout;
        edm_coeffs(levels[b], st->sigma_data, &cin, &cskip, &cout);
        launch_weighted_sum(st->d_gF + (size_t)b * d, grad_out + (size_t)b * d,
            grad_out + (size_t)b * d, cout, 0.0f, d, ctx->stream);
    }

    ctx_zero_cond_grad(ctx, batch);
    unet_backward(st->unet, st->d_scaled, st->d_F, st->d_gF, st->d_grad_in,
        batch, ctx);
    sequential_backward(st->embed, ctx->cond[st->cond_slot].grad, batch, ctx);
}

static void edm_parameters(Denoiser* self, ParamList* pl)
{
    EDMDenoiser* st = (EDMDenoiser*)self->state;
    sequential_parameters(st->embed, pl);
    unet_parameters(st->unet, pl);
}

static void edm_denoiser_free(Denoiser* self)
{
    EDMDenoiser* st = (EDMDenoiser*)self->state;
    unet_free(st->unet);
    sequential_free(st->embed);
    cudaFree(st->d_scaled);
    cudaFree(st->d_F);
    cudaFree(st->d_cnoise);
    cudaFree(st->d_gF);
    cudaFree(st->d_grad_in);
    free(st->h_cin);
    free(st->h_cskip);
    free(st->h_cout);
    free(st->h_cnoise);
    free(st);
    free(self);
}

Denoiser* edm_denoiser_create(Context* ctx, const UNetConfig* unet_cfg,
    int C, int H, int W, int max_batch, float sigma_data)
{
    Denoiser* self = (Denoiser*)malloc(sizeof(Denoiser));
    EDMDenoiser* st = (EDMDenoiser*)malloc(sizeof(EDMDenoiser));

    UNetConfig cfg = *unet_cfg;
    cfg.cond_slot = ctx_add_cond(ctx, cfg.emb_dim, max_batch);

    Module* emb_mods[] = {
        mp_fourier_create(ctx, cfg.emb_dim, 1.0f),
        mp_linear_create(ctx, cfg.emb_dim, cfg.emb_dim, 1.0f),
        activation_create(cfg.emb_dim, ACT_MP_SILU),
    };
    st->embed = sequential_create(emb_mods, 3, max_batch);
    st->unet = unet_create(ctx, &cfg, C, C, H, W, max_batch);

    st->cond_slot = cfg.cond_slot;
    st->img_dim = H * W * C;
    st->max_batch = max_batch;
    st->sigma_data = sigma_data;

    size_t n = (size_t)max_batch * st->img_dim;
    st->d_scaled = nn_device_alloc(n);
    st->d_F = nn_device_alloc(n);
    st->d_cnoise = nn_device_alloc((size_t)max_batch);
    st->d_gF = nn_device_alloc(n);
    st->d_grad_in = nn_device_alloc(n);
    st->h_cin = (float*)malloc((size_t)max_batch * sizeof(float));
    st->h_cskip = (float*)malloc((size_t)max_batch * sizeof(float));
    st->h_cout = (float*)malloc((size_t)max_batch * sizeof(float));
    st->h_cnoise = (float*)malloc((size_t)max_batch * sizeof(float));

    self->state = st;
    self->img_dim = st->img_dim;
    self->forward = edm_forward;
    self->backward = edm_backward;
    self->parameters = edm_parameters;
    self->free = edm_denoiser_free;
    return self;
}

/* ============================== training loss ============================== */

struct EDMLoss {
    EDMLossCfg cfg;
    int img_dim, max_batch;
    Loss* mse;
    float* d_noise;
    float* d_noisy;
    float* d_pred;
    float* d_grad;
    float* h_sigma;
    float* h_w;
};

EDMLoss* edm_loss_create(int img_dim, int max_batch, EDMLossCfg cfg)
{
    EDMLoss* l = (EDMLoss*)malloc(sizeof(EDMLoss));
    l->cfg = cfg;
    l->img_dim = img_dim;
    l->max_batch = max_batch;
    l->mse = loss_create(max_batch);
    size_t n = (size_t)max_batch * img_dim;
    l->d_noise = nn_device_alloc(n);
    l->d_noisy = nn_device_alloc(n);
    l->d_pred = nn_device_alloc(n);
    l->d_grad = nn_device_alloc(n);
    l->h_sigma = (float*)malloc((size_t)max_batch * sizeof(float));
    l->h_w = (float*)malloc((size_t)max_batch * sizeof(float));
    return l;
}

float edm_loss_step(void* loss, Denoiser* den, const float* d_x0,
    int batch, Context* ctx)
{
    EDMLoss* l = (EDMLoss*)loss;
    int d = l->img_dim;
    float sd = l->cfg.sigma_data;

    /* per-sample sigma ~ exp(N(P_mean, P_std)), clamped; EDM loss weight */
    for (int b = 0; b < batch; ++b) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float z = sqrtf(-2.0f * logf(u1)) * cosf(TWO_PI * u2);
        float sigma = expf(l->cfg.p_mean + l->cfg.p_std * z);
        sigma = fminf(fmaxf(sigma, l->cfg.sigma_min), l->cfg.sigma_max);
        l->h_sigma[b] = sigma;
        l->h_w[b] = (sigma * sigma + sd * sd) / ((sigma * sd) * (sigma * sd));
    }

    nn_fill_normal(ctx->curand, l->d_noise, (size_t)batch * d, 0.0f, 1.0f);
    for (int b = 0; b < batch; ++b)
        launch_weighted_sum(l->d_noisy + (size_t)b * d, d_x0 + (size_t)b * d,
            l->d_noise + (size_t)b * d, 1.0f, l->h_sigma[b], d, ctx->stream);

    den->forward(den, ctx, l->d_noisy, l->d_pred, batch, l->h_sigma);
    float mse = mse_weighted_forward_backward(l->mse, l->d_pred, d_x0,
        l->d_grad, batch, d, l->h_w, ctx);
    den->backward(den, ctx, l->d_noisy, l->d_grad, batch, l->h_sigma);
    return mse;
}

void edm_loss_free(EDMLoss* l)
{
    loss_free(l->mse);
    cudaFree(l->d_noise);
    cudaFree(l->d_noisy);
    cudaFree(l->d_pred);
    cudaFree(l->d_grad);
    free(l->h_sigma);
    free(l->h_w);
    free(l);
}
