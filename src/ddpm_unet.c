#include "nn.h"
#include <stdlib.h>
#include <string.h>

/* DDPM UNet with GroupNorm + additive time conditioning.

   Same encoder-decoder-skip topology as the EDM2 UNet, but uses conv_residual
   blocks (GroupNorm → SiLU → Conv → +emb → GroupNorm → SiLU → Conv → +skip)
   instead of EDM2's cond_conv_residual (mp_silu → Conv → FiLM → mp_silu → Conv
   → mp_sum).

   Stage numbering is identical to unet.c — see that file for the full map. */

struct DDPMUNet {
    int levels, max_batch;
    int C_in, C_out, H, W;
    int *ch, *Hs, *Ws;

    Module *in_conv, *out_conv;
    Module **enc_blocks, **down, **enc_proj;
    Module *mid_block1, *mid_attn, *mid_block2;
    Module **up, **dec_proj, **dec_blocks;

    int n_acts;
    int* act_sizes;
    float** acts;
    float** grads;
    float** skip_buf;
};

DDPMUNet* ddpm_unet_create(Context* ctx, DDPMUNetConfig* cfg, int C_in, int C_out,
    int H, int W, int max_batch)
{
    DDPMUNet* u = (DDPMUNet*)calloc(1, sizeof(DDPMUNet));
    int L = cfg->levels;
    NN_ASSERT(L >= 1, "ddpm_unet: need at least one level");
    NN_ASSERT(cfg->cond_slot >= 0, "ddpm_unet: cond_slot not assigned");
    NN_ASSERT(H % (1 << L) == 0 && W % (1 << L) == 0,
        "ddpm_unet: H and W must be divisible by 2^levels");
    for (int i = 0; i <= L; ++i)
        NN_ASSERT(cfg->channels[i] % cfg->num_groups == 0,
            "ddpm_unet: each channel count must be divisible by num_groups");
    u->levels = L;
    u->max_batch = max_batch;
    u->C_in = C_in;
    u->C_out = C_out;
    u->H = H;
    u->W = W;

    u->ch = (int*)malloc((L + 1) * sizeof(int));
    memcpy(u->ch, cfg->channels, (L + 1) * sizeof(int));
    u->Hs = (int*)malloc((L + 1) * sizeof(int));
    u->Ws = (int*)malloc((L + 1) * sizeof(int));
    u->Hs[0] = H;
    u->Ws[0] = W;
    for (int i = 1; i <= L; ++i) {
        u->Hs[i] = u->Hs[i - 1] / 2;
        u->Ws[i] = u->Ws[i - 1] / 2;
    }

    u->in_conv = conv2d_create(ctx, C_in, u->ch[0], 3, 1, 1, H, W, max_batch, 0, 1.0f);
    u->out_conv = conv2d_create(ctx, u->ch[0], C_out, 3, 1, 1, H, W, max_batch, 0, 1.0f);

    u->enc_blocks = (Module**)malloc(L * sizeof(Module*));
    u->down = (Module**)malloc(L * sizeof(Module*));
    u->enc_proj = (Module**)malloc(L * sizeof(Module*));
    u->up = (Module**)malloc(L * sizeof(Module*));
    u->dec_proj = (Module**)malloc(L * sizeof(Module*));
    u->dec_blocks = (Module**)malloc(L * sizeof(Module*));
    u->skip_buf = (float**)malloc(L * sizeof(float*));

    int G = cfg->num_groups;
    for (int i = 0; i < L; ++i) {
        int Hi = u->Hs[i], Wi = u->Ws[i], Ci = u->ch[i], Cn = u->ch[i + 1];
        u->enc_blocks[i] = conv_residual_create(ctx, Ci, cfg->emb_dim,
            Hi, Wi, max_batch, G, cfg->cond_slot);
        u->down[i] = downsample_create(Ci, Hi, Wi, max_batch);
        u->enc_proj[i] = conv2d_create(ctx, Ci, Cn, 1, 1, 0,
            Hi / 2, Wi / 2, max_batch, 0, 1.0f);
        u->up[i] = upsample_create(Cn, u->Hs[i + 1], u->Ws[i + 1], max_batch);
        u->dec_proj[i] = conv2d_create(ctx, Cn + Ci, Ci, 1, 1, 0,
            Hi, Wi, max_batch, 0, 1.0f);
        u->dec_blocks[i] = conv_residual_create(ctx, Ci, cfg->emb_dim,
            Hi, Wi, max_batch, G, cfg->cond_slot);
        u->skip_buf[i] = nn_device_alloc((size_t)max_batch * Hi * Wi * Ci);
    }

    int Hb = u->Hs[L], Wb = u->Ws[L], Cb = u->ch[L];
    u->mid_block1 = conv_residual_create(ctx, Cb, cfg->emb_dim,
        Hb, Wb, max_batch, G, cfg->cond_slot);
    u->mid_attn = attention_create(ctx, Cb, cfg->heads, Hb, Wb,
        max_batch, 0, 1.0f);
    u->mid_block2 = conv_residual_create(ctx, Cb, cfg->emb_dim,
        Hb, Wb, max_batch, G, cfg->cond_slot);

    int na = 5 + 7 * L;
    u->n_acts = na;
    u->act_sizes = (int*)malloc(na * sizeof(int));

    u->act_sizes[0] = H * W * C_in;
    u->act_sizes[1] = u->Hs[0] * u->Ws[0] * u->ch[0];
    for (int i = 0; i < L; ++i) {
        int Hi = u->Hs[i], Wi = u->Ws[i], Ci = u->ch[i], Cn = u->ch[i + 1];
        u->act_sizes[2 + 3 * i] = Hi * Wi * Ci;
        u->act_sizes[3 + 3 * i] = (Hi / 2) * (Wi / 2) * Ci;
        u->act_sizes[4 + 3 * i] = (Hi / 2) * (Wi / 2) * Cn;
    }
    int bot = 2 + 3 * L;
    u->act_sizes[bot] = u->act_sizes[bot + 1] = u->act_sizes[bot + 2] = Hb * Wb * Cb;
    int dec = 5 + 3 * L;
    for (int j = 0; j < L; ++j) {
        int i = L - 1 - j;
        int Hi = u->Hs[i], Wi = u->Ws[i], Ci = u->ch[i], Cn = u->ch[i + 1];
        u->act_sizes[dec + 4 * j] = Hi * Wi * Cn;
        u->act_sizes[dec + 4 * j + 1] = Hi * Wi * (Cn + Ci);
        u->act_sizes[dec + 4 * j + 2] = Hi * Wi * Ci;
        u->act_sizes[dec + 4 * j + 3] = Hi * Wi * Ci;
    }

    u->acts = (float**)malloc(na * sizeof(float*));
    u->grads = (float**)malloc(na * sizeof(float*));
    for (int s = 0; s < na; ++s) {
        u->acts[s] = nn_device_alloc((size_t)max_batch * u->act_sizes[s]);
        u->grads[s] = nn_device_alloc((size_t)max_batch * u->act_sizes[s]);
    }
    return u;
}

void ddpm_unet_forward(DDPMUNet* u, const float* in, float* out,
    int batch, Context* ctx)
{
    int L = u->levels;
    NN_CUDA_CHECK(cudaMemcpyAsync(u->acts[0], in,
        (size_t)batch * u->act_sizes[0] * sizeof(float),
        cudaMemcpyDeviceToDevice, ctx->stream));

    u->in_conv->forward(u->in_conv, u->acts[0], u->acts[1], batch, ctx);

    for (int i = 0; i < L; ++i) {
        int s = 2 + 3 * i;
        u->enc_blocks[i]->forward(u->enc_blocks[i],
            u->acts[s - 1], u->acts[s], batch, ctx);
        NN_CUDA_CHECK(cudaMemcpyAsync(u->skip_buf[i], u->acts[s],
            (size_t)batch * u->act_sizes[s] * sizeof(float),
            cudaMemcpyDeviceToDevice, ctx->stream));
        u->down[i]->forward(u->down[i], u->acts[s], u->acts[s + 1], batch, ctx);
        u->enc_proj[i]->forward(u->enc_proj[i],
            u->acts[s + 1], u->acts[s + 2], batch, ctx);
    }

    int bot = 2 + 3 * L;
    u->mid_block1->forward(u->mid_block1, u->acts[bot - 1], u->acts[bot], batch, ctx);
    u->mid_attn->forward(u->mid_attn, u->acts[bot], u->acts[bot + 1], batch, ctx);
    u->mid_block2->forward(u->mid_block2, u->acts[bot + 1], u->acts[bot + 2], batch, ctx);

    int dec = 5 + 3 * L;
    for (int j = 0; j < L; ++j) {
        int i = L - 1 - j;
        int s = dec + 4 * j;
        u->up[i]->forward(u->up[i], u->acts[s - 1], u->acts[s], batch, ctx);
        int spatial = batch * u->Hs[i] * u->Ws[i];
        launch_channel_cat(u->acts[s + 1], u->acts[s], u->skip_buf[i],
            spatial, u->ch[i + 1], u->ch[i], ctx->stream);
        u->dec_proj[i]->forward(u->dec_proj[i],
            u->acts[s + 1], u->acts[s + 2], batch, ctx);
        u->dec_blocks[i]->forward(u->dec_blocks[i],
            u->acts[s + 2], u->acts[s + 3], batch, ctx);
    }

    u->out_conv->forward(u->out_conv, u->acts[u->n_acts - 1], out, batch, ctx);
}

void ddpm_unet_backward(DDPMUNet* u, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)out;
    int L = u->levels;
    int na = u->n_acts;
    int bot = 2 + 3 * L;
    int dec = 5 + 3 * L;

    u->out_conv->backward(u->out_conv, u->acts[na - 1], NULL,
        grad_out, u->grads[na - 1], batch, ctx);

    for (int level = 0; level < L; ++level) {
        int j_fwd = L - 1 - level;
        int s = dec + 4 * j_fwd;

        u->dec_blocks[level]->backward(u->dec_blocks[level],
            u->acts[s + 2], u->acts[s + 3],
            u->grads[s + 3], u->grads[s + 2], batch, ctx);
        u->dec_proj[level]->backward(u->dec_proj[level],
            u->acts[s + 1], u->acts[s + 2],
            u->grads[s + 2], u->grads[s + 1], batch, ctx);

        int spatial = batch * u->Hs[level] * u->Ws[level];
        launch_channel_split(u->grads[s], u->skip_buf[level],
            u->grads[s + 1], spatial,
            u->ch[level + 1], u->ch[level], ctx->stream);

        u->up[level]->backward(u->up[level],
            u->acts[s - 1], u->acts[s],
            u->grads[s], u->grads[s - 1], batch, ctx);
    }

    u->mid_block2->backward(u->mid_block2,
        u->acts[bot + 1], u->acts[bot + 2],
        u->grads[bot + 2], u->grads[bot + 1], batch, ctx);
    u->mid_attn->backward(u->mid_attn,
        u->acts[bot], u->acts[bot + 1],
        u->grads[bot + 1], u->grads[bot], batch, ctx);
    u->mid_block1->backward(u->mid_block1,
        u->acts[bot - 1], u->acts[bot],
        u->grads[bot], u->grads[bot - 1], batch, ctx);

    for (int i = L - 1; i >= 0; --i) {
        int s = 2 + 3 * i;

        u->enc_proj[i]->backward(u->enc_proj[i],
            u->acts[s + 1], u->acts[s + 2],
            u->grads[s + 2], u->grads[s + 1], batch, ctx);
        u->down[i]->backward(u->down[i],
            u->acts[s], u->acts[s + 1],
            u->grads[s + 1], u->grads[s], batch, ctx);

        launch_weighted_sum(u->grads[s], u->grads[s], u->skip_buf[i],
            1.0f, 1.0f, batch * u->act_sizes[s], ctx->stream);

        u->enc_blocks[i]->backward(u->enc_blocks[i],
            u->acts[s - 1], u->acts[s],
            u->grads[s], u->grads[s - 1], batch, ctx);
    }

    u->in_conv->backward(u->in_conv, u->acts[0], u->acts[1],
        u->grads[1], grad_in, batch, ctx);
}

void ddpm_unet_parameters(DDPMUNet* u, ParamList* pl)
{
    int L = u->levels;
    u->in_conv->parameters(u->in_conv, pl);
    for (int i = 0; i < L; ++i) {
        u->enc_blocks[i]->parameters(u->enc_blocks[i], pl);
        u->enc_proj[i]->parameters(u->enc_proj[i], pl);
    }
    u->mid_block1->parameters(u->mid_block1, pl);
    u->mid_attn->parameters(u->mid_attn, pl);
    u->mid_block2->parameters(u->mid_block2, pl);
    for (int i = L - 1; i >= 0; --i) {
        u->dec_proj[i]->parameters(u->dec_proj[i], pl);
        u->dec_blocks[i]->parameters(u->dec_blocks[i], pl);
    }
    u->out_conv->parameters(u->out_conv, pl);
}

void ddpm_unet_free(DDPMUNet* u)
{
    int L = u->levels;
    u->in_conv->free(u->in_conv);
    u->out_conv->free(u->out_conv);
    for (int i = 0; i < L; ++i) {
        u->enc_blocks[i]->free(u->enc_blocks[i]);
        u->down[i]->free(u->down[i]);
        u->enc_proj[i]->free(u->enc_proj[i]);
        u->up[i]->free(u->up[i]);
        u->dec_proj[i]->free(u->dec_proj[i]);
        u->dec_blocks[i]->free(u->dec_blocks[i]);
        cudaFree(u->skip_buf[i]);
    }
    u->mid_block1->free(u->mid_block1);
    u->mid_attn->free(u->mid_attn);
    u->mid_block2->free(u->mid_block2);
    for (int s = 0; s < u->n_acts; ++s) {
        cudaFree(u->acts[s]);
        cudaFree(u->grads[s]);
    }
    free(u->enc_blocks);
    free(u->down);
    free(u->enc_proj);
    free(u->up);
    free(u->dec_proj);
    free(u->dec_blocks);
    free(u->skip_buf);
    free(u->acts);
    free(u->grads);
    free(u->act_sizes);
    free(u->ch);
    free(u->Hs);
    free(u->Ws);
    free(u);
}
