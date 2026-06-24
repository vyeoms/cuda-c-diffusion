/* Finite-difference check of the hand-written module backwards: compares each
 * analytic gradient to a central difference of L = <forward(in), g> for a fixed
 * random cotangent g. Add a module by calling check_module().
 *
 * Two subtleties: (1) we run cuBLAS in true FP32 -- TF32's ~10-bit mantissa would
 * swamp the perturbation; (2) magnitude-preserving layers normalize their weights
 * in place and their backward returns dL/dW_raw on the unit sphere, so we warm up
 * once, snapshot, then restore before each numeric forward. */

#include "nn.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define K_SAMPLES 20 /* perturbed elements sampled per tensor */
#define EPS 1.0e-3f
/* Absolute floor for the pass test. Near-zero true gradients (e.g. a key bias,
   which softmax shift-invariance makes a no-op) leave the central difference as
   pure FP32 noise ~1e-3; an absolute floor stops the relative metric from
   blowing up there, while real backward bugs differ by O(0.1-1) and still fail. */
#define ATOL 1.0e-2

static Context g_ctx;

/* ----------------------------- host RNG ----------------------------------- */

static float randn(void)
{
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* ------------------------- small device helpers --------------------------- */

static void d2h(float* h, const float* d, int n)
{
    NN_CUDA_CHECK(cudaMemcpy(h, d, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
}
static void h2d(float* d, const float* h, int n)
{
    NN_CUDA_CHECK(cudaMemcpy(d, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
}
static double dot_host(const float* a, const float* b, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += (double)a[i] * (double)b[i];
    return s;
}
/* Violation ratio under a combined absolute+relative tolerance: <=1 passes.
   |num - ana| <= ATOL + rtol*(|num| + |ana|). */
static double viol_ratio(double num, double ana, double rtol)
{
    return fabs(num - ana) / (ATOL + rtol * (fabs(num) + fabs(ana)));
}

/* --------------------------- context lifecycle ---------------------------- */

static void ctx_init(unsigned long long seed)
{
    NN_CUBLAS_CHECK(cublasCreate(&g_ctx.cublas));
    /* deliberately NOT CUBLAS_TF32_TENSOR_OP_MATH: we need full FP32 here */
    g_ctx.stream = 0;
    NN_CURAND_CHECK(curandCreateGenerator(&g_ctx.curand, CURAND_RNG_PSEUDO_DEFAULT));
    NN_CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(g_ctx.curand, seed));
    NN_CURAND_CHECK(curandSetStream(g_ctx.curand, g_ctx.stream));
    g_ctx.n_cond = 0;
}
static void ctx_destroy(void)
{
    curandDestroyGenerator(g_ctx.curand);
    cublasDestroy(g_ctx.cublas);
}

/* --------------------------- parameter snapshot --------------------------- */
/* Host-side copy of every parameter tensor, captured on the unit sphere after a
   warm-up forward and rewritten to the device before each numeric forward. */

typedef struct {
    float* host;
    float* dev; /* aliases the live param buffer */
    int size;
} Snap;

static void snap_capture(const ParamList* pl, Snap* snaps)
{
    for (int k = 0; k < pl->count; ++k) {
        snaps[k].size = pl->items[k].size;
        snaps[k].dev = pl->items[k].param;
        snaps[k].host = (float*)malloc((size_t)snaps[k].size * sizeof(float));
        d2h(snaps[k].host, snaps[k].dev, snaps[k].size);
    }
}
static void snap_restore_all(const ParamList* pl, const Snap* snaps)
{
    for (int k = 0; k < pl->count; ++k)
        h2d(snaps[k].dev, snaps[k].host, snaps[k].size);
}
static void snap_free(const ParamList* pl, Snap* snaps)
{
    for (int k = 0; k < pl->count; ++k)
        free(snaps[k].host);
}

/* Choose <=K_SAMPLES distinct indices in [0,size). Small sizes -> take all. */
static int pick_indices(int size, int* idx)
{
    if (size <= K_SAMPLES) {
        for (int i = 0; i < size; ++i)
            idx[i] = i;
        return size;
    }
    for (int i = 0; i < K_SAMPLES; ++i)
        idx[i] = rand() % size;
    return K_SAMPLES;
}

/* --------------------------------- core ----------------------------------- */

static int check_module(const char* tag, Module* m, int batch, double rtol)
{
    int in = m->in_dim, out = m->out_dim;
    int N_in = batch * in, N_out = batch * out;

    float* d_in = nn_device_alloc(N_in);
    float* d_out = nn_device_alloc(N_out);
    float* d_gout = nn_device_alloc(N_out);
    float* d_gin = nn_device_alloc(N_in);

    float* h_in = (float*)malloc((size_t)N_in * sizeof(float));
    float* h_out = (float*)malloc((size_t)N_out * sizeof(float));
    float* h_g = (float*)malloc((size_t)N_out * sizeof(float));
    float* h_gin = (float*)malloc((size_t)N_in * sizeof(float));

    for (int i = 0; i < N_in; ++i)
        h_in[i] = randn();
    for (int i = 0; i < N_out; ++i)
        h_g[i] = randn();
    h2d(d_in, h_in, N_in);
    h2d(d_gout, h_g, N_out);

    ParamList pl;
    param_list_init(&pl);
    if (m->parameters)
        m->parameters(m, &pl);

    /* warm-up forward: projects any in-place-normalized weights onto the sphere */
    m->forward(m, d_in, d_out, batch, &g_ctx);
    NN_CUDA_CHECK(cudaDeviceSynchronize());

    Snap* snaps = pl.count ? (Snap*)malloc(pl.count * sizeof(Snap)) : NULL;
    snap_capture(&pl, snaps);

    /* analytic gradients at the snapshot point */
    for (int k = 0; k < pl.count; ++k)
        NN_CUDA_CHECK(cudaMemset(pl.items[k].grad, 0,
            (size_t)pl.items[k].size * sizeof(float)));
    m->forward(m, d_in, d_out, batch, &g_ctx);
    m->backward(m, d_in, d_out, d_gout, d_gin, batch, &g_ctx);
    NN_CUDA_CHECK(cudaDeviceSynchronize());
    d2h(h_gin, d_gin, N_in);

    /* stash analytic param grads on the host before numeric forwards clobber them */
    float** ana_pg = pl.count ? (float**)malloc(pl.count * sizeof(float*)) : NULL;
    for (int k = 0; k < pl.count; ++k) {
        ana_pg[k] = (float*)malloc((size_t)pl.items[k].size * sizeof(float));
        d2h(ana_pg[k], pl.items[k].grad, pl.items[k].size);
    }

    double worst = 0.0;
    const char* worst_where = "input";
    int worst_k = -1, worst_j = -1;
    double worst_num = 0.0, worst_ana = 0.0;
    int idx[K_SAMPLES];

    /* ---- gradient w.r.t. the input ---- */
    {
        int n = pick_indices(in * batch, idx);
        for (int s = 0; s < n; ++s) {
            int j = idx[s];
            float save = h_in[j];

            snap_restore_all(&pl, snaps);
            h_in[j] = save + EPS;
            h2d(d_in, h_in, N_in);
            m->forward(m, d_in, d_out, batch, &g_ctx);
            d2h(h_out, d_out, N_out);
            double Lp = dot_host(h_out, h_g, N_out);

            snap_restore_all(&pl, snaps);
            h_in[j] = save - EPS;
            h2d(d_in, h_in, N_in);
            m->forward(m, d_in, d_out, batch, &g_ctx);
            d2h(h_out, d_out, N_out);
            double Lm = dot_host(h_out, h_g, N_out);

            h_in[j] = save;
            double num = (Lp - Lm) / (2.0 * EPS);
            double e = viol_ratio(num, (double)h_gin[j], rtol);
            if (e > worst) {
                worst = e;
                worst_where = "input";
                worst_k = -1;
                worst_j = j;
                worst_num = num;
                worst_ana = h_gin[j];
            }
        }
        h2d(d_in, h_in, N_in);
    }

    /* ---- gradient w.r.t. each parameter tensor ---- */
    for (int k = 0; k < pl.count; ++k) {
        int sz = pl.items[k].size;
        float* dev = pl.items[k].param;
        int n = pick_indices(sz, idx);
        for (int s = 0; s < n; ++s) {
            int j = idx[s];
            float base = snaps[k].host[j];

            snap_restore_all(&pl, snaps);
            float vp = base + EPS;
            h2d(dev + j, &vp, 1);
            m->forward(m, d_in, d_out, batch, &g_ctx);
            d2h(h_out, d_out, N_out);
            double Lp = dot_host(h_out, h_g, N_out);

            snap_restore_all(&pl, snaps);
            float vm = base - EPS;
            h2d(dev + j, &vm, 1);
            m->forward(m, d_in, d_out, batch, &g_ctx);
            d2h(h_out, d_out, N_out);
            double Lm = dot_host(h_out, h_g, N_out);

            double num = (Lp - Lm) / (2.0 * EPS);
            double e = viol_ratio(num, (double)ana_pg[k][j], rtol);
            if (e > worst) {
                worst = e;
                worst_where = "param";
                worst_k = k;
                worst_j = j;
                worst_num = num;
                worst_ana = ana_pg[k][j];
            }
        }
    }
    snap_restore_all(&pl, snaps);

    int pass = worst <= 1.0;
    printf("  [%-4s] %-22s  worst_ratio=%.2f  (%s",
        pass ? "PASS" : "FAIL", tag, worst, worst_where);
    if (worst_k >= 0)
        printf(" #%d", worst_k);
    printf(")%s\n", pass ? "" : "   <-- CHECK BACKWARD");
    if (!pass)
        printf("         at idx %d: numeric=%+.6e  analytic=%+.6e\n",
            worst_j, worst_num, worst_ana);

    for (int k = 0; k < pl.count; ++k)
        free(ana_pg[k]);
    free(ana_pg);
    snap_free(&pl, snaps);
    free(snaps);
    param_list_free(&pl);
    m->free(m);
    free(h_in);
    free(h_out);
    free(h_g);
    free(h_gin);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_gout);
    cudaFree(d_gin);
    return pass ? 0 : 1;
}

/* --------------------------------- main ----------------------------------- */

int main(void)
{
    srand(1234);
    ctx_init(42ULL);
    int B = 2;
    int fails = 0;

    printf("Gradient check (central difference, eps=%.0e, FP32 cuBLAS)\n", (double)EPS);

    printf("\n-- dense --\n");
    fails += check_module("linear", linear_create(&g_ctx, 12, 8), B, 2e-2);
    fails += check_module("mp_linear", mp_linear_create(&g_ctx, 12, 8, 1.0f), B, 2e-2);

    printf("\n-- activations --\n");
    fails += check_module("act_silu", activation_create(8, ACT_SILU), B, 2e-2);
    fails += check_module("act_gelu", activation_create(8, ACT_GELU), B, 2e-2);
    fails += check_module("act_tanh", activation_create(8, ACT_TANH), B, 2e-2);
    fails += check_module("act_mp_silu", activation_create(8, ACT_MP_SILU), B, 2e-2);

    printf("\n-- normalization --\n");
    fails += check_module("layernorm", layernorm_create(16, B), B, 2e-2);
    fails += check_module("groupnorm", spatial_groupnorm_create(8, 4, 6, 6, B), B, 2e-2);

    printf("\n-- convolution (NHWC) --\n");
    fails += check_module("conv3x3", conv2d_create(&g_ctx, 3, 4, 3, 1, 1, 6, 6, B, 0, 1.0f), B, 2e-2);
    fails += check_module("conv3x3_mp", conv2d_create(&g_ctx, 3, 4, 3, 1, 1, 6, 6, B, 1, 1.0f), B, 2e-2);
    fails += check_module("conv1x1_mp", conv2d_create(&g_ctx, 4, 6, 1, 1, 0, 5, 5, B, 1, 1.0f), B, 2e-2);

    printf("\n-- resampling --\n");
    fails += check_module("downsample", downsample_create(4, 6, 6, B), B, 2e-2);
    fails += check_module("upsample", upsample_create(4, 6, 6, B), B, 2e-2);

    printf("\n-- attention --\n");
    fails += check_module("attention", attention_create(&g_ctx, 8, 2, 4, 4, B, 0, 1.0f), B, 3e-2);
    fails += check_module("attention_mp", attention_create(&g_ctx, 8, 2, 4, 4, B, 1, 1.0f), B, 3e-2);

    ctx_destroy();
    printf("\n%s (%d failing module%s)\n",
        fails ? "GRADCHECK FAILED" : "ALL GRADIENTS OK",
        fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
