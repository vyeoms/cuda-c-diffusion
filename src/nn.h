#ifndef NN_H
#define NN_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ======================= small shared utilities ======================= */

#define NN_CUDA_CHECK(call)                                         \
    do {                                                            \
        cudaError_t _e = (call);                                    \
        if (_e != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                            \
            exit(1);                                                \
        }                                                           \
    } while (0)

#define NN_CUBLAS_CHECK(call)                                                \
    do {                                                                     \
        cublasStatus_t _s = (call);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS %s:%d: status %d\n", __FILE__, __LINE__, \
                (int)_s);                                                    \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define NN_CURAND_CHECK(call)                                                \
    do {                                                                     \
        curandStatus_t _r = (call);                                          \
        if (_r != CURAND_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuRAND %s:%d: status %d\n", __FILE__, __LINE__, \
                (int)_r);                                                    \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static inline float* nn_device_alloc(size_t n_floats)
{
    float* p;
    NN_CUDA_CHECK(cudaMalloc((void**)&p, n_floats * sizeof(float)));
    return p;
}

/* ============================== context ============================== */

#define CTX_MAX_COND 4

typedef struct {
    const float* fwd;
    float* grad;
    int dim;
} CondSlot;

typedef struct {
    cublasHandle_t cublas;
    curandGenerator_t curand;
    cudaStream_t stream;
    CondSlot cond[CTX_MAX_COND];
    int n_cond;
} Context;

static inline int ctx_add_cond(Context* ctx, int dim, int max_batch)
{
    if (ctx->n_cond >= CTX_MAX_COND) {
        fprintf(stderr, "ctx_add_cond: too many conditioning slots\n");
        exit(1);
    }
    int slot = ctx->n_cond++;
    ctx->cond[slot].dim = dim;
    ctx->cond[slot].fwd = NULL;
    ctx->cond[slot].grad = nn_device_alloc((size_t)max_batch * dim);
    return slot;
}

static inline void ctx_set_cond(Context* ctx, int slot, const float* fwd)
{
    ctx->cond[slot].fwd = fwd;
}

static inline void ctx_zero_cond_grad(Context* ctx, int batch)
{
    for (int i = 0; i < ctx->n_cond; ++i)
        if (ctx->cond[i].grad)
            NN_CUDA_CHECK(cudaMemsetAsync(ctx->cond[i].grad, 0,
                (size_t)batch * ctx->cond[i].dim * sizeof(float), ctx->stream));
}

static inline void ctx_free_cond(Context* ctx)
{
    for (int i = 0; i < ctx->n_cond; ++i) {
        if (ctx->cond[i].grad) {
            cudaFree(ctx->cond[i].grad);
            ctx->cond[i].grad = NULL;
        }
        ctx->cond[i].fwd = NULL;
    }
    ctx->n_cond = 0;
}

/* Fill a device buffer with N(mean, std). curandGenerateNormal needs an even
   count (it produces Box-Muller pairs), so for odd n we fill the even prefix
   and patch the final element from one extra pair. */
static inline void nn_fill_normal(curandGenerator_t g, float* dst, size_t n,
    float mean, float std)
{
    if ((n & 1u) == 0) {
        NN_CURAND_CHECK(curandGenerateNormal(g, dst, n, mean, std));
    } else {
        if (n > 1)
            NN_CURAND_CHECK(curandGenerateNormal(g, dst, n - 1, mean, std));
        float* pair;
        NN_CUDA_CHECK(cudaMalloc((void**)&pair, 2 * sizeof(float)));
        NN_CURAND_CHECK(curandGenerateNormal(g, pair, 2, mean, std));
        NN_CUDA_CHECK(cudaMemcpy(dst + (n - 1), pair, sizeof(float),
            cudaMemcpyDeviceToDevice));
        cudaFree(pair);
    }
}

/* ===================== parameters / optimizer view ===================== */

typedef struct {
    float* param; /* device */
    float* grad; /* device */
    int size; /* element count */
} Param;

typedef struct {
    Param* items;
    int count;
    int capacity;
} ParamList;

void param_list_init(ParamList* pl);
void param_list_add(ParamList* pl, float* param, float* grad, int size);
void param_list_free(ParamList* pl);

/* ========================== module interface ==========================
   The container owns inter-module buffers and passes the same `in`/`out`
   pointers on backward that were used on forward.                         */

typedef struct Module Module;
struct Module {
    const char* name;
    int in_dim, out_dim;
    void* state; /* module-specific params/buffers */

    void (*forward)(Module* self, const float* in, float* out,
        int batch, Context* ctx);

    void (*backward)(Module* self, const float* in, const float* out,
        const float* grad_out, float* grad_in,
        int batch, Context* ctx);

    void (*parameters)(Module* self, ParamList* pl); /* may be NULL-effect */
    void (*free)(Module* self);
};

/* ============================== modules ============================== */

typedef enum { ACT_IDENTITY = 0,
    ACT_RELU,
    ACT_GELU,
    ACT_SILU,
    ACT_TANH,
    ACT_MP_SILU } Activation;

Module* linear_create(Context* ctx, int in_dim, int out_dim);
Module* activation_create(int dim, Activation act); /* ACT_MP_SILU = magnitude-preserving */
Module* mp_linear_create(Context* ctx, int in_dim, int out_dim, float gain); /* forced weight norm, no bias */
Module* residual_create(Module** branch, int n, int max_batch, float t);
/* out = mp_sum(x, branch(x), t): magnitude-preserving residual; branch maps dim->dim */
Module* mp_fourier_create(Context* ctx, int num_channels, float bandwidth); /* scalar -> [batch,C] features */
Module* cond_residual_create(Context* ctx, int dim, int emb_dim, int max_batch, float t,
    int cond_slot);

/* --- non-MP / DDPM building blocks --- */
Module* sinusoidal_create(int num_channels, float max_period); /* deterministic timestep embedding */
Module* learned_embed_create(Context* ctx, int num_embeddings, int dim); /* table lookup; index given as float */
Module* plain_residual_create(Module** branch, int n, int max_batch); /* x + branch(x) (no magnitude preservation) */
Module* layernorm_create(int dim, int max_batch); /* per-row LayerNorm with affine gamma/beta */

/* ========================= sequential container ========================= */

typedef struct Sequential Sequential;

Sequential* sequential_create(Module** modules, int n, int max_batch);
void sequential_forward(Sequential* s, const float* input,
    int batch, Context* ctx);
void sequential_backward(Sequential* s, const float* grad_output,
    int batch, Context* ctx);
float* sequential_output(Sequential* s); /* final activations (device) */
float* sequential_input_grad(Sequential* s); /* grad wrt input (device) */
void sequential_parameters(Sequential* s, ParamList* pl);
void sequential_free(Sequential* s);

/* ============================== optimizer ============================== */

typedef struct Optimizer Optimizer;

Optimizer* optimizer_sgd(const ParamList* pl, float lr);
Optimizer* optimizer_adam(const ParamList* pl, float lr,
    float beta1, float beta2, float eps);
void optimizer_step(Optimizer* o, const ParamList* pl, Context* ctx);
void optimizer_set_lr(Optimizer* o, float lr);
void optimizer_free(Optimizer* o);

/* Karras et al. inverse-sqrt LR schedule:
   linear warmup over t_warmup steps, then 1/sqrt(t) decay after t_ref steps. */
static inline float lr_inv_sqrt(float lr_ref, int step, int t_warmup, int t_ref)
{
    float warmup = (t_warmup > 0 && step < t_warmup)
        ? (float)step / (float)t_warmup
        : 1.0f;
    float decay = (step > t_ref)
        ? sqrtf((float)t_ref / (float)step)
        : 1.0f;
    return lr_ref * warmup * decay;
}

/* ================================= EMA ================================= */

typedef struct EMA EMA;

EMA* ema_create(const ParamList* pl, float decay);
void ema_update(EMA* ema, const ParamList* pl, Context* ctx);
void ema_swap(EMA* ema, ParamList* pl, Context* ctx);
void ema_free(EMA* ema);

/* ============================ checkpointing ============================ */

int checkpoint_save(const char* path, const ParamList* pl);
int checkpoint_load(const char* path, ParamList* pl);

/* ========================= spatial modules (NHWC) ========================= */

Module* conv2d_create(Context* ctx, int C_in, int C_out, int K, int stride, int pad,
    int H, int W, int max_batch, int mp_mode, float gain);
Module* downsample_create(int C, int H, int W, int max_batch);
Module* upsample_create(int C, int H, int W, int max_batch);
Module* attention_create(Context* ctx, int C, int heads, int H, int W,
    int max_batch, int mp_mode, float gain);
Module* cond_conv_residual_create(Context* ctx, int C, int emb_dim, int H, int W,
    int max_batch, float t, int cond_slot);
Module* spatial_groupnorm_create(int C, int num_groups, int H, int W, int max_batch);
Module* conv_residual_create(Context* ctx, int C, int emb_dim, int H, int W,
    int max_batch, int num_groups, int cond_slot);

/* ============================= UNet container ============================= */

typedef struct {
    int levels;
    int* channels;
    int emb_dim;
    int heads;
    float t;
    int mp_mode;
    int cond_slot;
} UNetConfig;

typedef struct UNet UNet;

UNet* unet_create(Context* ctx, UNetConfig* cfg, int C_in, int C_out,
    int H, int W, int max_batch);
void unet_forward(UNet* u, const float* in, float* out, int batch, Context* ctx);
void unet_backward(UNet* u, const float* in, const float* out,
    const float* grad_out, float* grad_in, int batch, Context* ctx);
void unet_parameters(UNet* u, ParamList* pl);
void unet_free(UNet* u);

/* ========================= DDPM UNet container ========================= */

typedef struct {
    int levels;
    int* channels;
    int emb_dim;
    int heads;
    int num_groups;
    int cond_slot;
} DDPMUNetConfig;

typedef struct DDPMUNet DDPMUNet;

DDPMUNet* ddpm_unet_create(Context* ctx, DDPMUNetConfig* cfg, int C_in, int C_out,
    int H, int W, int max_batch);
void ddpm_unet_forward(DDPMUNet* u, const float* in, float* out,
    int batch, Context* ctx);
void ddpm_unet_backward(DDPMUNet* u, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx);
void ddpm_unet_parameters(DDPMUNet* u, ParamList* pl);
void ddpm_unet_free(DDPMUNet* u);

/* ===================== loss head (softmax + x-entropy) ===================== */

typedef struct Loss Loss;

Loss* loss_create(int max_batch);
Loss* loss_softmax_xent_create(int num_classes, int max_batch);
/* writes grad wrt logits into grad_logits; returns mean loss over the batch */
float loss_forward_backward(Loss* l, const float* logits, const int* targets,
    float* grad_logits, int batch, Context* ctx);
/* MSE loss for regression (float targets, dim = total elements per sample) */
float mse_forward_backward(Loss* l, const float* pred, const float* target,
    float* grad, int batch, int dim, Context* ctx);
/* weighted MSE: per-sample weights (host array, length=batch) scale both the
   reported loss and the gradient. */
float mse_weighted_forward_backward(Loss* l, const float* pred, const float* target,
    float* grad, int batch, int dim,
    const float* sample_weights, Context* ctx);
void loss_free(Loss* l);

/* ================= device kernel launchers (kernels.cu, C ABI) ================= */

#ifdef __cplusplus
extern "C" {
#endif

void launch_add_bias(float* Y, const float* b, int M, int N, cudaStream_t s);

void launch_activation_forward(float* out, const float* in, int n,
    Activation act, cudaStream_t s);
void launch_activation_backward(float* grad_in, const float* grad_out,
    const float* in, int n, Activation act,
    cudaStream_t s);

void launch_softmax_xent(float* grad_logits, float* loss_row, const float* logits,
    const int* targets, int batch, int C, cudaStream_t s);

void launch_bias_grad(float* gb, const float* grad_out, int batch, int N,
    cudaStream_t s);

void launch_sgd_step(float* p, const float* g, float lr, int n, cudaStream_t s);
void launch_adam_step(float* p, float* m, float* v, const float* g,
    float lr, float beta1, float beta2, float eps,
    float bc1, float bc2, int n, cudaStream_t s);

/* magnitude-preserving ops */
void launch_weight_normalize(float* W, int in, int out, float eps, cudaStream_t s);
void launch_weight_normalize_backward(float* gW, const float* Wn, const float* gain,
    float* g_gain, int in, int out, cudaStream_t s);
void launch_scale_by_scalar(float* x, const float* s, int n, cudaStream_t st);
void launch_mp_sum(float* out, const float* a, const float* b, float t, int n,
    cudaStream_t s);
void launch_weighted_sum(float* out, const float* a, const float* b,
    float wa, float wb, int n, cudaStream_t s);
void launch_mp_fourier(float* out, const float* x, const float* freqs,
    const float* phases, int batch, int C, cudaStream_t s);
void launch_modulate(float* h2, const float* h1, const float* c, int n, cudaStream_t s);
void launch_modulate_backward(float* gh1, float* gc, const float* gh2,
    const float* h1, const float* c, int n, cudaStream_t s);
void launch_sinusoidal(float* out, const float* t, int batch, int C,
    float log_max_period, cudaStream_t s);
void launch_embed_gather(float* out, const float* table, const float* idx,
    int batch, int C, int T, cudaStream_t s);
void launch_embed_scatter(float* grad_table, const float* grad_out, const float* idx,
    int batch, int C, int T, cudaStream_t s);
void launch_layernorm_forward(float* y, float* mu, float* rstd, const float* x,
    const float* gamma, const float* beta,
    int batch, int C, float eps, cudaStream_t s);
void launch_layernorm_backward(float* grad_in, float* ggamma, float* gbeta,
    const float* grad_out, const float* x, const float* gamma,
    const float* mu, const float* rstd,
    int batch, int C, cudaStream_t s);
void launch_groupnorm_forward(float* y, float* mu, float* rstd,
    const float* x, const float* gamma, const float* beta,
    int batch, int HW, int C, int G,
    float eps, cudaStream_t s);
void launch_groupnorm_backward(float* dx, float* ggamma, float* gbeta,
    const float* dy, const float* x, const float* gamma,
    const float* mu, const float* rstd,
    int batch, int HW, int C, int G, cudaStream_t s);
void launch_mp_cat(float* out, const float* a, const float* b, float t,
    int batch, int Na, int Nb, cudaStream_t s);

/* convolution (NHWC) */
void launch_im2col(float* col, const float* im, int batch, int H, int W, int C,
    int K, int stride, int pad, int H_out, int W_out, cudaStream_t s);
void launch_col2im(float* grad_im, const float* grad_col, int batch, int H, int W, int C,
    int K, int stride, int pad, int H_out, int W_out, cudaStream_t s);

/* spatial pooling / upsampling (NHWC) */
void launch_avg_pool_2x(float* out, const float* in, int batch, int H, int W, int C,
    cudaStream_t s);
void launch_avg_pool_2x_backward(float* grad_in, const float* grad_out,
    int batch, int H, int W, int C, cudaStream_t s);
void launch_nearest_upsample_2x(float* out, const float* in,
    int batch, int H, int W, int C, cudaStream_t s);
void launch_nearest_upsample_2x_backward(float* grad_in, const float* grad_out,
    int batch, int H, int W, int C, cudaStream_t s);

/* attention */
void launch_attn_softmax(float* out, const float* in, int N, int S, cudaStream_t s);
void launch_attn_softmax_backward(float* grad_in, const float* grad_out,
    const float* p, int N, int S, cudaStream_t s);
void launch_permute_0213(float* out, const float* in, int B, int d1, int d2, int d3,
    cudaStream_t s);

/* spatial FiLM (NHWC) */
void launch_spatial_modulate(float* out, const float* in, const float* cond,
    int batch, int H, int W, int C, cudaStream_t s);
void launch_spatial_modulate_backward(float* grad_in, float* grad_cond,
    const float* grad_out, const float* in,
    const float* cond,
    int batch, int H, int W, int C, cudaStream_t s);

/* spatial add / reduce (NHWC) */
void launch_spatial_add(float* inout, const float* bias,
    int batch, int H, int W, int C, cudaStream_t s);
void launch_spatial_reduce_sum(float* out, const float* in,
    int batch, int HW, int C, cudaStream_t s);

/* channel concat / split (NHWC) */
void launch_channel_cat(float* out, const float* a, const float* b,
    int spatial, int Ca, int Cb, cudaStream_t s);
void launch_channel_split(float* grad_a, float* grad_b, const float* grad_out,
    int spatial, int Ca, int Cb, cudaStream_t s);
void launch_mp_channel_cat(float* out, const float* a, const float* b,
    float t, int spatial, int Ca, int Cb, cudaStream_t s);
void launch_mp_channel_split(float* grad_a, float* grad_b, const float* grad_out,
    float t, int spatial, int Ca, int Cb, cudaStream_t s);

/* MSE loss */
void launch_mse_fwd_bwd(float* grad, float* loss_row, const float* pred,
    const float* target, int batch, int dim, cudaStream_t s);

void launch_clamp(float* x, float lo, float hi, int n, cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif /* NN_H */
