#include "nn.h"
#include <math.h>

/* ---------- fast approximations ---------- */

__device__ __forceinline__ float fast_tanh(float x)
{
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return fmaxf(-1.0f, fminf(1.0f, a / b));
}

__device__ __forceinline__ float fast_sigmoid(float x)
{
    return fmaxf(0.0f, fminf(1.0f, (fast_tanh(x * 0.5f) + 1.0f) * 0.5f));
}

/* ---------- device activation helpers ---------- */

__device__ float act_apply(float v, Activation act)
{
    switch (act) {
    case ACT_RELU:
        return v > 0.0f ? v : 0.0f;
    case ACT_GELU:
        return 0.5f * v * (1.0f + fast_tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    case ACT_SILU:
        return v * fast_sigmoid(v);
    case ACT_MP_SILU:
        return v * fast_sigmoid(v) * (1.0f / 0.596f);
    case ACT_TANH:
        return fast_tanh(v);
    case ACT_IDENTITY:
    default:
        return v;
    }
}

__device__ float act_grad(float z, Activation act)
{
    switch (act) {
    case ACT_RELU:
        return z > 0.0f ? 1.0f : 0.0f;
    case ACT_TANH: {
        float t = fast_tanh(z);
        return 1.0f - t * t;
    }
    case ACT_SILU: {
        float s = fast_sigmoid(z);
        return s * (1.0f + z * (1.0f - s));
    }
    case ACT_MP_SILU: {
        float s = fast_sigmoid(z);
        return s * (1.0f + z * (1.0f - s)) * (1.0f / 0.596f);
    }
    case ACT_GELU: {
        float c = 0.7978845608f;
        float u = c * (z + 0.044715f * z * z * z);
        float t = fast_tanh(u);
        float du = c * (1.0f + 3.0f * 0.044715f * z * z);
        return 0.5f * (1.0f + t) + 0.5f * z * (1.0f - t * t) * du;
    }
    case ACT_IDENTITY:
    default:
        return 1.0f;
    }
}

/* ---------- block-reduce helper ---------- */

__device__ float block_reduce_sum(float val)
{
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    if (lane == 0)
        smem[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0)
            smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

/* ---------- kernels ---------- */

__global__ void add_bias_kernel(float* Y, const float* b, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x, total = M * N;
    for (; idx < total; idx += stride)
        Y[idx] += b[idx % N];
}

__global__ void activation_forward_kernel(float* out, const float* in,
    int n, Activation act)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        out[i] = act_apply(in[i], act);
}

__global__ void activation_backward_kernel(float* grad_in, const float* grad_out,
    const float* in, int n, Activation act)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        grad_in[i] = grad_out[i] * act_grad(in[i], act);
}

__global__ void softmax_xent_kernel(float* grad_logits, float* loss_row,
    const float* logits, const int* targets,
    int batch, int C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch)
        return;

    const float* z = logits + row * C;
    float* g = grad_logits + row * C;
    int t = targets[row];

    float m = z[0];
    for (int j = 1; j < C; ++j)
        if (z[j] > m)
            m = z[j];
    float sum = 0.0f;
    for (int j = 0; j < C; ++j)
        sum += __expf(z[j] - m);

    loss_row[row] = (m + logf(sum)) - z[t];

    float inv = 1.0f / (float)batch;
    for (int j = 0; j < C; ++j) {
        float p = __expf(z[j] - m) / sum;
        g[j] = (p - (j == t ? 1.0f : 0.0f)) * inv;
    }
}

__global__ void bias_grad_kernel(float* gb, const float* grad_out, int batch, int N)
{
    int lane = threadIdx.x & 31;
    int n = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (n >= N)
        return;
    float s = 0.0f;
    for (int r = lane; r < batch; r += 32)
        s += grad_out[r * N + n];
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);
    if (lane == 0)
        gb[n] = s;
}

__global__ void sgd_step_kernel(float* p, const float* g, float lr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        p[i] -= lr * g[i];
}

__global__ void adam_step_kernel(float* p, float* m, float* v, const float* g,
    float lr, float b1, float b2, float eps,
    float bc1, float bc2, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float gi = g[i];
        float mi = b1 * m[i] + (1.0f - b1) * gi;
        float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
    }
}

/* ---------- launchers ---------- */

static int grid_for(int n, int block)
{
    int g = (n + block - 1) / block;
    return g > 1024 ? 1024 : g;
}

extern "C" void launch_add_bias(float* Y, const float* b, int M, int N, cudaStream_t s)
{
    int block = 256;
    add_bias_kernel<<<grid_for(M * N, block), block, 0, s>>>(Y, b, M, N);
}

extern "C" void launch_activation_forward(float* out, const float* in, int n,
    Activation act, cudaStream_t s)
{
    int block = 256;
    activation_forward_kernel<<<grid_for(n, block), block, 0, s>>>(out, in, n, act);
}

extern "C" void launch_activation_backward(float* grad_in, const float* grad_out,
    const float* in, int n, Activation act,
    cudaStream_t s)
{
    int block = 256;
    activation_backward_kernel<<<grid_for(n, block), block, 0, s>>>(grad_in, grad_out,
        in, n, act);
}

extern "C" void launch_softmax_xent(float* grad_logits, float* loss_row,
    const float* logits, const int* targets,
    int batch, int C, cudaStream_t s)
{
    int block = 128;
    int grid = (batch + block - 1) / block;
    softmax_xent_kernel<<<grid, block, 0, s>>>(grad_logits, loss_row, logits,
        targets, batch, C);
}

extern "C" void launch_bias_grad(float* gb, const float* grad_out, int batch, int N,
    cudaStream_t s)
{
    int warps_per_block = 4;
    int block = warps_per_block * 32;
    int grid = (N + warps_per_block - 1) / warps_per_block;
    bias_grad_kernel<<<grid, block, 0, s>>>(gb, grad_out, batch, N);
}

extern "C" void launch_sgd_step(float* p, const float* g, float lr, int n,
    cudaStream_t s)
{
    int block = 256;
    sgd_step_kernel<<<grid_for(n, block), block, 0, s>>>(p, g, lr, n);
}

extern "C" void launch_adam_step(float* p, float* m, float* v, const float* g,
    float lr, float b1, float b2, float eps,
    float bc1, float bc2, int n, cudaStream_t s)
{
    int block = 256;
    adam_step_kernel<<<grid_for(n, block), block, 0, s>>>(p, m, v, g, lr, b1, b2,
        eps, bc1, bc2, n);
}

/* ===================== magnitude-preserving (MP) ops ===================== */

/* Forced weight normalization: scale each output column of W to unit L2 norm
   over fan-in. W is [in, out] row-major, so consecutive threads (consecutive
   output columns) read consecutive addresses -> coalesced. In place. */
__global__ void weight_normalize_kernel(float* W, int in, int out, float eps)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= out)
        return;
    float ss = 0.0f;
    for (int i = lane; i < in; i += 32) {
        float w = W[i * out + j];
        ss += w * w;
    }
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xffffffff, ss, off);
    float inv = rsqrtf(__shfl_sync(0xffffffff, ss, 0) + eps);
    for (int i = lane; i < in; i += 32)
        W[i * out + j] *= inv;
}

/* Fused weight-norm backward per column j:
   dot_j = <Wn_j, M_j>,  g_gain += dot_j,
   gW_j = gain * (M_j - dot_j * Wn_j)    (tangential projection + gain) */
__global__ void weight_normalize_backward_kernel(float* gW, const float* Wn,
    const float* gain, float* g_gain,
    int in, int out)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= out)
        return;
    float dot = 0.0f;
    for (int i = lane; i < in; i += 32)
        dot += Wn[i * out + j] * gW[i * out + j];
    for (int off = 16; off > 0; off >>= 1)
        dot += __shfl_down_sync(0xffffffff, dot, off);
    dot = __shfl_sync(0xffffffff, dot, 0);
    if (lane == 0)
        atomicAdd(g_gain, dot);
    float g = gain[0];
    for (int i = lane; i < in; i += 32)
        gW[i * out + j] = g * (gW[i * out + j] - dot * Wn[i * out + j]);
}

/* x[i] *= s[0] */
__global__ void scale_by_scalar_kernel(float* x, const float* s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    float v = s[0];
    for (; i < n; i += stride)
        x[i] *= v;
}

__global__ void weighted_sum_kernel(float* out, const float* a, const float* b,
    float wa, float wb, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        out[i] = wa * a[i] + wb * b[i];
}

/* mp_cat: magnitude-preserving concat of a[batch,Na] and b[batch,Nb] into
   out[batch,Na+Nb]; the per-branch scales are folded into the placement copy. */
__global__ void mp_cat_kernel(float* out, const float* a, const float* b,
    float wa, float wb, int batch, int Na, int Nb)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    int N = Na + Nb, total = batch * N;
    for (; idx < total; idx += stride) {
        int row = idx / N, col = idx % N;
        out[idx] = (col < Na) ? wa * a[row * Na + col]
                              : wb * b[row * Nb + (col - Na)];
    }
}

extern "C" void launch_weight_normalize(float* W, int in, int out, float eps,
    cudaStream_t s)
{
    int warps = 4, threads = warps * 32;
    weight_normalize_kernel<<<(out + warps - 1) / warps, threads, 0, s>>>(W, in, out, eps);
}

extern "C" void launch_weight_normalize_backward(float* gW, const float* Wn,
    const float* gain, float* g_gain,
    int in, int out, cudaStream_t s)
{
    int warps = 4, threads = warps * 32;
    weight_normalize_backward_kernel<<<(out + warps - 1) / warps, threads, 0, s>>>(
        gW, Wn, gain, g_gain, in, out);
}

extern "C" void launch_scale_by_scalar(float* x, const float* s, int n,
    cudaStream_t st)
{
    int block = 256;
    scale_by_scalar_kernel<<<grid_for(n, block), block, 0, st>>>(x, s, n);
}

extern "C" void launch_weighted_sum(float* out, const float* a, const float* b,
    float wa, float wb, int n, cudaStream_t s)
{
    int block = 256;
    weighted_sum_kernel<<<grid_for(n, block), block, 0, s>>>(out, a, b, wa, wb, n);
}

extern "C" void launch_mp_sum(float* out, const float* a, const float* b,
    float t, int n, cudaStream_t s)
{
    float d = sqrtf((1.0f - t) * (1.0f - t) + t * t);
    float wa = (1.0f - t) / d, wb = t / d;
    launch_weighted_sum(out, a, b, wa, wb, n, s);
}

extern "C" void launch_mp_cat(float* out, const float* a, const float* b,
    float t, int batch, int Na, int Nb, cudaStream_t s)
{
    float denom = (1.0f - t) * (1.0f - t) + t * t;
    float C = sqrtf((float)(Na + Nb) / denom);
    float wa = C / sqrtf((float)Na) * (1.0f - t);
    float wb = C / sqrtf((float)Nb) * t;
    int block = 256;
    mp_cat_kernel<<<grid_for(batch * (Na + Nb), block), block, 0, s>>>(
        out, a, b, wa, wb, batch, Na, Nb);
}

/* ===================== embedding / conditioning ops ===================== */

/* MPFourier: out[b,k] = sqrt(2) * cos(2*pi*(x[b]*freqs[k] + phases[k])). The
   sqrt(2) makes E[out^2]=1 since E[cos^2]=1/2. freqs ~ N(0,bandwidth), phases
   ~ U[0,1); x is one scalar per sample (the noise level). */
__global__ void mp_fourier_kernel(float* out, const float* x, const float* freqs,
    const float* phases, int batch, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    int total = batch * C;
    for (; idx < total; idx += stride) {
        int b = idx / C, k = idx % C;
        out[idx] = 1.41421356237f * cosf(6.28318530718f * (x[b] * freqs[k] + phases[k]));
    }
}

/* FiLM modulation: h2 = h1 * (c + 1). At init (c=0) this is the identity. */
__global__ void modulate_kernel(float* h2, const float* h1, const float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        h2[i] = h1[i] * (c[i] + 1.0f);
}

/* modulation backward: gh1 = gh2*(c+1);  gc = gh2*h1 */
__global__ void modulate_backward_kernel(float* gh1, float* gc, const float* gh2,
    const float* h1, const float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float g = gh2[i];
        gh1[i] = g * (c[i] + 1.0f);
        gc[i] = g * h1[i];
    }
}

extern "C" void launch_mp_fourier(float* out, const float* x, const float* freqs,
    const float* phases, int batch, int C,
    cudaStream_t s)
{
    int block = 256;
    mp_fourier_kernel<<<grid_for(batch * C, block), block, 0, s>>>(
        out, x, freqs, phases, batch, C);
}

extern "C" void launch_modulate(float* h2, const float* h1, const float* c, int n,
    cudaStream_t s)
{
    int block = 256;
    modulate_kernel<<<grid_for(n, block), block, 0, s>>>(h2, h1, c, n);
}

extern "C" void launch_modulate_backward(float* gh1, float* gc, const float* gh2,
    const float* h1, const float* c, int n,
    cudaStream_t s)
{
    int block = 256;
    modulate_backward_kernel<<<grid_for(n, block), block, 0, s>>>(gh1, gc, gh2, h1, c, n);
}

/* ===================== alternative embeddings ===================== */

/* Sinusoidal timestep embedding: out = [sin(t·f), cos(t·f)] with
   geometrically spaced f_j = exp(-log_max_period · j/half). */
__global__ void sinusoidal_kernel(float* out, const float* t, int batch, int C,
    float log_max_period)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    int total = batch * C, half = C / 2;
    for (; idx < total; idx += stride) {
        int b = idx / C, c = idx % C;
        int j = (c < half) ? c : (c - half);
        float freq = expf(-log_max_period * (float)j / (float)half);
        float arg = t[b] * freq;
        out[idx] = (c < half) ? sinf(arg) : cosf(arg);
    }
}

/* Learned embedding table: gather row idx[b] of table[T,C] (idx given as float). */
__global__ void embed_gather_kernel(float* out, const float* table, const float* idx,
    int batch, int C, int T)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    int total = batch * C;
    for (; i < total; i += stride) {
        int b = i / C, c = i % C;
        int t = (int)idx[b];
        t = t < 0 ? 0 : (t >= T ? T - 1 : t);
        out[i] = table[t * C + c];
    }
}

/* Learned embedding backward: scatter-add grad_out into the gathered rows.
   atomicAdd guards against duplicate indices within the batch. */
__global__ void embed_scatter_kernel(float* grad_table, const float* grad_out,
    const float* idx, int batch, int C, int T)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    int total = batch * C;
    for (; i < total; i += stride) {
        int b = i / C, c = i % C;
        int t = (int)idx[b];
        t = t < 0 ? 0 : (t >= T ? T - 1 : t);
        atomicAdd(&grad_table[t * C + c], grad_out[i]);
    }
}

/* ===================== layer normalization ===================== */

/* y = gamma * (x - mean) / sqrt(var + eps) + beta.
   One warp per row; caches mean and rstd for backward. */
__global__ void layernorm_forward_kernel(float* y, float* mu, float* rstd,
    const float* x, const float* gamma,
    const float* beta, int batch, int C, float eps)
{
    int b = blockIdx.x;
    if (b >= batch)
        return;
    const float* xb = x + (size_t)b * C;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        sum += xb[c];
    float m = block_reduce_sum(sum) / (float)C;
    float vsum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float d = xb[c] - m;
        vsum += d * d;
    }
    float rs = rsqrtf(block_reduce_sum(vsum) / (float)C + eps);
    if (threadIdx.x == 0) {
        mu[b] = m;
        rstd[b] = rs;
    }
    float* yb = y + (size_t)b * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        yb[c] = gamma[c] * ((xb[c] - m) * rs) + beta[c];
}

/* dL/dx for layernorm: dx_i = rstd*(g_i - mean(g) - xhat_i*mean(g*xhat)),
   g_i = grad_out_i * gamma_i, means over the feature axis. */
__global__ void layernorm_backward_x_kernel(float* grad_in, const float* grad_out,
    const float* x, const float* gamma,
    const float* mu, const float* rstd,
    int batch, int C)
{
    int b = blockIdx.x;
    if (b >= batch)
        return;
    const float* xb = x + (size_t)b * C;
    const float* gob = grad_out + (size_t)b * C;
    float m = mu[b], rs = rstd[b];
    float pg = 0.0f, pgx = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float xhat = (xb[c] - m) * rs;
        float gv = gob[c] * gamma[c];
        pg += gv;
        pgx += gv * xhat;
    }
    float sum_g = block_reduce_sum(pg);
    float sum_gx = block_reduce_sum(pgx);
    float invC = 1.0f / (float)C;
    float* gib = grad_in + (size_t)b * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float xhat = (xb[c] - m) * rs;
        float gv = gob[c] * gamma[c];
        gib[c] = rs * (gv - sum_g * invC - xhat * sum_gx * invC);
    }
}

/* dL/dgamma and dL/dbeta: reduce over the batch for each channel. */
__global__ void layernorm_backward_param_kernel(float* ggamma, float* gbeta,
    const float* grad_out, const float* x,
    const float* mu, const float* rstd,
    int batch, int C)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float gg = 0.0f, gb = 0.0f;
    for (int b = 0; b < batch; ++b) {
        float xhat = (x[(size_t)b * C + c] - mu[b]) * rstd[b];
        float go = grad_out[(size_t)b * C + c];
        gg += go * xhat;
        gb += go;
    }
    ggamma[c] = gg;
    gbeta[c] = gb;
}

extern "C" void launch_sinusoidal(float* out, const float* t, int batch, int C,
    float log_max_period, cudaStream_t s)
{
    int block = 256;
    sinusoidal_kernel<<<grid_for(batch * C, block), block, 0, s>>>(
        out, t, batch, C, log_max_period);
}

extern "C" void launch_embed_gather(float* out, const float* table, const float* idx,
    int batch, int C, int T, cudaStream_t s)
{
    int block = 256;
    embed_gather_kernel<<<grid_for(batch * C, block), block, 0, s>>>(
        out, table, idx, batch, C, T);
}

extern "C" void launch_embed_scatter(float* grad_table, const float* grad_out,
    const float* idx, int batch, int C, int T,
    cudaStream_t s)
{
    int block = 256;
    embed_scatter_kernel<<<grid_for(batch * C, block), block, 0, s>>>(
        grad_table, grad_out, idx, batch, C, T);
}

extern "C" void launch_layernorm_forward(float* y, float* mu, float* rstd,
    const float* x, const float* gamma,
    const float* beta, int batch, int C,
    float eps, cudaStream_t s)
{
    int block = 256;
    layernorm_forward_kernel<<<batch, block, 0, s>>>(
        y, mu, rstd, x, gamma, beta, batch, C, eps);
}

extern "C" void launch_layernorm_backward(float* grad_in, float* ggamma, float* gbeta,
    const float* grad_out, const float* x,
    const float* gamma, const float* mu,
    const float* rstd, int batch, int C,
    cudaStream_t s)
{
    int block = 256;
    layernorm_backward_x_kernel<<<batch, block, 0, s>>>(
        grad_in, grad_out, x, gamma, mu, rstd, batch, C);
    int blk2 = 128;
    layernorm_backward_param_kernel<<<(C + blk2 - 1) / blk2, blk2, 0, s>>>(
        ggamma, gbeta, grad_out, x, mu, rstd, batch, C);
}

/* ===================== spatial group normalization (NHWC) ===================== */

/* y = gamma * (x - mean) / sqrt(var + eps) + beta, grouped over (H*W, C/G).
   One warp per (batch, group); total_groups = batch * G. */
__global__ void groupnorm_forward_kernel(float* y, float* mu, float* rstd,
    const float* x, const float* gamma,
    const float* beta,
    int total_groups, int HW, int C, int G,
    float eps)
{
    int idx = blockIdx.x;
    if (idx >= total_groups)
        return;

    int b = idx / G, g = idx % G;
    int cpg = C / G;
    int n = HW * cpg;
    int c0 = g * cpg;

    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int hw = i / cpg, cl = i % cpg;
        float val = x[b * HW * C + hw * C + c0 + cl];
        sum += val;
        sum_sq += val * val;
    }
    float mean = block_reduce_sum(sum) / (float)n;
    float var = block_reduce_sum(sum_sq) / (float)n - mean * mean;
    float rs = rsqrtf(var + eps);
    if (threadIdx.x == 0) {
        mu[idx] = mean;
        rstd[idx] = rs;
    }

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int hw = i / cpg, cl = i % cpg;
        int off = b * HW * C + hw * C + c0 + cl;
        y[off] = gamma[c0 + cl] * ((x[off] - mean) * rs) + beta[c0 + cl];
    }
}

__global__ void groupnorm_backward_x_kernel(float* dx, const float* dy,
    const float* x, const float* gamma,
    const float* mu, const float* rstd,
    int total_groups, int HW, int C, int G)
{
    int idx = blockIdx.x;
    if (idx >= total_groups)
        return;

    int b = idx / G, g = idx % G;
    int cpg = C / G;
    int n = HW * cpg;
    int c0 = g * cpg;
    float mean = mu[idx], rs = rstd[idx];

    float pg = 0.0f, pgx = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int hw = i / cpg, cl = i % cpg;
        int off = b * HW * C + hw * C + c0 + cl;
        float xhat = (x[off] - mean) * rs;
        float gv = dy[off] * gamma[c0 + cl];
        pg += gv;
        pgx += gv * xhat;
    }
    float sum_g = block_reduce_sum(pg);
    float sum_gx = block_reduce_sum(pgx);

    float inv_n = 1.0f / (float)n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int hw = i / cpg, cl = i % cpg;
        int off = b * HW * C + hw * C + c0 + cl;
        float xhat = (x[off] - mean) * rs;
        float gv = dy[off] * gamma[c0 + cl];
        dx[off] = rs * (gv - sum_g * inv_n - xhat * sum_gx * inv_n);
    }
}

__global__ void groupnorm_backward_param_kernel(float* ggamma, float* gbeta,
    const float* dy, const float* x,
    const float* mu, const float* rstd,
    int B, int HW, int C, int G)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    int g = c / (C / G);
    float gg = 0.0f, gb = 0.0f;
    for (int b = 0; b < B; ++b) {
        float mean = mu[b * G + g], rs = rstd[b * G + g];
        for (int hw = 0; hw < HW; ++hw) {
            int off = b * HW * C + hw * C + c;
            float xhat = (x[off] - mean) * rs;
            gg += dy[off] * xhat;
            gb += dy[off];
        }
    }
    ggamma[c] = gg;
    gbeta[c] = gb;
}

extern "C" void launch_groupnorm_forward(float* y, float* mu, float* rstd,
    const float* x, const float* gamma,
    const float* beta,
    int batch, int HW, int C, int G,
    float eps, cudaStream_t s)
{
    int total = batch * G;
    int block = 256;
    groupnorm_forward_kernel<<<total, block, 0, s>>>(
        y, mu, rstd, x, gamma, beta, total, HW, C, G, eps);
}

extern "C" void launch_groupnorm_backward(float* dx, float* ggamma, float* gbeta,
    const float* dy, const float* x,
    const float* gamma, const float* mu,
    const float* rstd,
    int batch, int HW, int C, int G,
    cudaStream_t s)
{
    int total = batch * G;
    int block = 256;
    groupnorm_backward_x_kernel<<<total, block, 0, s>>>(
        dx, dy, x, gamma, mu, rstd, total, HW, C, G);
    int blk2 = 128;
    groupnorm_backward_param_kernel<<<(C + blk2 - 1) / blk2, blk2, 0, s>>>(
        ggamma, gbeta, dy, x, mu, rstd, batch, HW, C, G);
}

/* ===================== convolution ops (NHWC layout) ===================== */

/* im2col: [batch, H, W, C] → col[batch*H_out*W_out, K*K*C] */
__global__ void im2col_kernel(float* col, const float* im,
    int batch, int H, int W, int C,
    int K, int stride, int pad,
    int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int KKC = K * K * C;
    int total = batch * H_out * W_out * KKC;
    for (; idx < total; idx += st) {
        int tmp = idx;
        int ci = tmp % C;
        tmp /= C;
        int kw = tmp % K;
        tmp /= K;
        int kh = tmp % K;
        tmp /= K;
        int ow = tmp % W_out;
        tmp /= W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int b = tmp;
        int ih = oh * stride + kh - pad;
        int iw = ow * stride + kw - pad;
        float val = 0.0f;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
            val = im[((b * H + ih) * W + iw) * C + ci];
        int col_row = (b * H_out + oh) * W_out + ow;
        int col_col = (kh * K + kw) * C + ci;
        col[col_row * KKC + col_col] = val;
    }
}

/* col2im: scatter-add from grad_col back to grad_input (NHWC).
   grad_input must be zeroed before calling. */
__global__ void col2im_kernel(float* grad_im, const float* grad_col,
    int batch, int H, int W, int C,
    int K, int stride, int pad,
    int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int KKC = K * K * C;
    int total = batch * H_out * W_out * KKC;
    for (; idx < total; idx += st) {
        int tmp = idx;
        int ci = tmp % C;
        tmp /= C;
        int kw = tmp % K;
        tmp /= K;
        int kh = tmp % K;
        tmp /= K;
        int ow = tmp % W_out;
        tmp /= W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int b = tmp;
        int ih = oh * stride + kh - pad;
        int iw = ow * stride + kw - pad;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            int col_row = (b * H_out + oh) * W_out + ow;
            int col_col = (kh * K + kw) * C + ci;
            atomicAdd(&grad_im[((b * H + ih) * W + iw) * C + ci],
                grad_col[col_row * KKC + col_col]);
        }
    }
}

extern "C" void launch_im2col(float* col, const float* im,
    int batch, int H, int W, int C,
    int K, int stride, int pad,
    int H_out, int W_out, cudaStream_t s)
{
    int block = 256;
    int total = batch * H_out * W_out * K * K * C;
    im2col_kernel<<<grid_for(total, block), block, 0, s>>>(
        col, im, batch, H, W, C, K, stride, pad, H_out, W_out);
}

extern "C" void launch_col2im(float* grad_im, const float* grad_col,
    int batch, int H, int W, int C,
    int K, int stride, int pad,
    int H_out, int W_out, cudaStream_t s)
{
    int block = 256;
    int total = batch * H_out * W_out * K * K * C;
    col2im_kernel<<<grid_for(total, block), block, 0, s>>>(
        grad_im, grad_col, batch, H, W, C, K, stride, pad, H_out, W_out);
}

/* ===================== spatial pooling / upsampling (NHWC) ===================== */

/* 2x average pooling: [B,H,W,C] -> [B,H/2,W/2,C] */
__global__ void avg_pool_2x_kernel(float* out, const float* in,
    int batch, int H, int W, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int Ho = H / 2, Wo = W / 2;
    int total = batch * Ho * Wo * C;
    for (; idx < total; idx += st) {
        int c = idx % C;
        int tmp = idx / C;
        int ow = tmp % Wo;
        tmp /= Wo;
        int oh = tmp % Ho;
        tmp /= Ho;
        int b = tmp;
        int ih = oh * 2, iw = ow * 2;
        out[idx] = 0.25f * (in[((b * H + ih) * W + iw) * C + c] + in[((b * H + ih) * W + iw + 1) * C + c] + in[((b * H + ih + 1) * W + iw) * C + c] + in[((b * H + ih + 1) * W + iw + 1) * C + c]);
    }
}

/* No overlapping writes — non-overlapping 2x2 blocks, no atomicAdd needed. */
__global__ void avg_pool_2x_backward_kernel(float* grad_in, const float* grad_out,
    int batch, int H, int W, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int Ho = H / 2, Wo = W / 2;
    int total = batch * Ho * Wo * C;
    for (; idx < total; idx += st) {
        int c = idx % C;
        int tmp = idx / C;
        int ow = tmp % Wo;
        tmp /= Wo;
        int oh = tmp % Ho;
        tmp /= Ho;
        int b = tmp;
        int ih = oh * 2, iw = ow * 2;
        float g = grad_out[idx] * 0.25f;
        grad_in[((b * H + ih) * W + iw) * C + c] = g;
        grad_in[((b * H + ih) * W + iw + 1) * C + c] = g;
        grad_in[((b * H + ih + 1) * W + iw) * C + c] = g;
        grad_in[((b * H + ih + 1) * W + iw + 1) * C + c] = g;
    }
}

extern "C" void launch_avg_pool_2x(float* out, const float* in,
    int batch, int H, int W, int C, cudaStream_t s)
{
    int block = 256;
    avg_pool_2x_kernel<<<grid_for(batch * (H / 2) * (W / 2) * C, block), block, 0, s>>>(
        out, in, batch, H, W, C);
}

extern "C" void launch_avg_pool_2x_backward(float* grad_in, const float* grad_out,
    int batch, int H, int W, int C,
    cudaStream_t s)
{
    int block = 256;
    avg_pool_2x_backward_kernel<<<grid_for(batch * (H / 2) * (W / 2) * C, block), block, 0, s>>>(
        grad_in, grad_out, batch, H, W, C);
}

/* 2x nearest-neighbor upsampling: [B,H,W,C] -> [B,2H,2W,C] */
__global__ void nearest_upsample_2x_kernel(float* out, const float* in,
    int batch, int H, int W, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int Ho = 2 * H, Wo = 2 * W;
    int total = batch * Ho * Wo * C;
    for (; idx < total; idx += st) {
        int c = idx % C;
        int tmp = idx / C;
        int ow = tmp % Wo;
        tmp /= Wo;
        int oh = tmp % Ho;
        tmp /= Ho;
        int b = tmp;
        out[idx] = in[((b * H + oh / 2) * W + ow / 2) * C + c];
    }
}

/* Backward of nearest upsample: sum 2x2 block of grad_out into one grad_in pixel */
__global__ void nearest_upsample_2x_backward_kernel(float* grad_in, const float* grad_out,
    int batch, int H, int W, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int Ho = 2 * H, Wo = 2 * W;
    int total = batch * H * W * C;
    for (; idx < total; idx += st) {
        int c = idx % C;
        int tmp = idx / C;
        int iw = tmp % W;
        tmp /= W;
        int ih = tmp % H;
        tmp /= H;
        int b = tmp;
        int oh = ih * 2, ow = iw * 2;
        grad_in[idx] = grad_out[((b * Ho + oh) * Wo + ow) * C + c]
            + grad_out[((b * Ho + oh) * Wo + ow + 1) * C + c]
            + grad_out[((b * Ho + oh + 1) * Wo + ow) * C + c]
            + grad_out[((b * Ho + oh + 1) * Wo + ow + 1) * C + c];
    }
}

extern "C" void launch_nearest_upsample_2x(float* out, const float* in,
    int batch, int H, int W, int C,
    cudaStream_t s)
{
    int block = 256;
    nearest_upsample_2x_kernel<<<grid_for(batch * 4 * H * W * C, block), block, 0, s>>>(
        out, in, batch, H, W, C);
}

extern "C" void launch_nearest_upsample_2x_backward(float* grad_in, const float* grad_out,
    int batch, int H, int W, int C,
    cudaStream_t s)
{
    int block = 256;
    nearest_upsample_2x_backward_kernel<<<grid_for(batch * H * W * C, block), block, 0, s>>>(
        grad_in, grad_out, batch, H, W, C);
}

/* ===================== attention ops ===================== */

/* Per-row softmax for attention scores. One thread per row (S is small). */
__global__ void attn_softmax_kernel(float* out, const float* in, int N, int S)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;
    const float* x = in + row * S;
    float* y = out + row * S;
    float m = x[0];
    for (int j = 1; j < S; ++j)
        if (x[j] > m)
            m = x[j];
    float sum = 0.0f;
    for (int j = 0; j < S; ++j) {
        y[j] = __expf(x[j] - m);
        sum += y[j];
    }
    float inv = 1.0f / sum;
    for (int j = 0; j < S; ++j)
        y[j] *= inv;
}

/* Backward: grad_in = p * (grad_out - dot(p, grad_out)) */
__global__ void attn_softmax_backward_kernel(float* grad_in, const float* grad_out,
    const float* p, int N, int S)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N)
        return;
    const float* go = grad_out + row * S;
    const float* pr = p + row * S;
    float* gi = grad_in + row * S;
    float dot = 0.0f;
    for (int j = 0; j < S; ++j)
        dot += go[j] * pr[j];
    for (int j = 0; j < S; ++j)
        gi[j] = pr[j] * (go[j] - dot);
}

extern "C" void launch_attn_softmax(float* out, const float* in,
    int N, int S, cudaStream_t s)
{
    int block = 128;
    attn_softmax_kernel<<<(N + block - 1) / block, block, 0, s>>>(out, in, N, S);
}

extern "C" void launch_attn_softmax_backward(float* grad_in, const float* grad_out,
    const float* p, int N, int S,
    cudaStream_t s)
{
    int block = 128;
    attn_softmax_backward_kernel<<<(N + block - 1) / block, block, 0, s>>>(
        grad_in, grad_out, p, N, S);
}

/* Permute [B, d1, d2, d3] -> [B, d2, d1, d3] (swap middle two dims).
   For attention: [B,S,H,D]->[B,H,S,D]. Inverse: swap d1,d2 args. */
__global__ void permute_0213_kernel(float* out, const float* in,
    int B, int d1, int d2, int d3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int total = B * d1 * d2 * d3;
    for (; idx < total; idx += st) {
        int tmp = idx;
        int j3 = tmp % d3;
        tmp /= d3;
        int j2 = tmp % d2;
        tmp /= d2;
        int j1 = tmp % d1;
        tmp /= d1;
        int b = tmp;
        out[((b * d2 + j2) * d1 + j1) * d3 + j3] = in[idx];
    }
}

extern "C" void launch_permute_0213(float* out, const float* in,
    int B, int d1, int d2, int d3, cudaStream_t s)
{
    int block = 256;
    permute_0213_kernel<<<grid_for(B * d1 * d2 * d3, block), block, 0, s>>>(
        out, in, B, d1, d2, d3);
}

/* ===================== spatial FiLM (NHWC) ===================== */

/* out[b,h,w,c] = in[b,h,w,c] * (cond[b,c] + 1) */
__global__ void spatial_modulate_kernel(float* out, const float* in,
    const float* cond,
    int batch, int H, int W, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int total = batch * H * W * C;
    for (; idx < total; idx += st) {
        int c = idx % C;
        int b = idx / (H * W * C);
        out[idx] = in[idx] * (cond[b * C + c] + 1.0f);
    }
}

/* grad_in[b,h,w,c] = grad_out[b,h,w,c] * (cond[b,c] + 1) */
__global__ void spatial_modulate_backward_in_kernel(float* grad_in,
    const float* grad_out,
    const float* cond,
    int batch, int H, int W, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int total = batch * H * W * C;
    for (; idx < total; idx += st) {
        int c = idx % C;
        int b = idx / (H * W * C);
        grad_in[idx] = grad_out[idx] * (cond[b * C + c] + 1.0f);
    }
}

/* grad_cond[b,c] = sum_{h,w} grad_out[b,h,w,c] * in[b,h,w,c] */
__global__ void spatial_modulate_backward_cond_kernel(float* grad_cond,
    const float* grad_out,
    const float* in,
    int batch, int H, int W, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int total = batch * C;
    for (; idx < total; idx += st) {
        int c = idx % C;
        int b = idx / C;
        float sum = 0.0f;
        for (int hw = 0; hw < H * W; ++hw)
            sum += grad_out[(b * H * W + hw) * C + c] * in[(b * H * W + hw) * C + c];
        grad_cond[idx] = sum;
    }
}

extern "C" void launch_spatial_modulate(float* out, const float* in,
    const float* cond,
    int batch, int H, int W, int C,
    cudaStream_t s)
{
    int block = 256;
    spatial_modulate_kernel<<<grid_for(batch * H * W * C, block), block, 0, s>>>(
        out, in, cond, batch, H, W, C);
}

extern "C" void launch_spatial_modulate_backward(float* grad_in, float* grad_cond,
    const float* grad_out,
    const float* in, const float* cond,
    int batch, int H, int W, int C,
    cudaStream_t s)
{
    int block = 256;
    int total_spatial = batch * H * W * C;
    spatial_modulate_backward_in_kernel<<<grid_for(total_spatial, block), block, 0, s>>>(
        grad_in, grad_out, cond, batch, H, W, C);
    spatial_modulate_backward_cond_kernel<<<grid_for(batch * C, block), block, 0, s>>>(
        grad_cond, grad_out, in, batch, H, W, C);
}

/* ===================== spatial add / reduce (NHWC) ===================== */

/* inout[b,h,w,c] += bias[b,c]  (broadcast over H,W) */
__global__ void spatial_add_kernel(float* inout, const float* bias,
    int total, int HWC, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    for (; idx < total; idx += st) {
        int c = idx % C;
        int b = idx / HWC;
        inout[idx] += bias[b * C + c];
    }
}

/* out[b,c] = sum_{h,w} in[b,h,w,c] */
__global__ void spatial_reduce_sum_kernel(float* out, const float* in,
    int total, int HW, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    for (; idx < total; idx += st) {
        int c = idx % C;
        int b = idx / C;
        float sum = 0.0f;
        for (int hw = 0; hw < HW; ++hw)
            sum += in[b * HW * C + hw * C + c];
        out[idx] = sum;
    }
}

extern "C" void launch_spatial_add(float* inout, const float* bias,
    int batch, int H, int W, int C,
    cudaStream_t s)
{
    int total = batch * H * W * C;
    int block = 256;
    spatial_add_kernel<<<grid_for(total, block), block, 0, s>>>(
        inout, bias, total, H * W * C, C);
}

extern "C" void launch_spatial_reduce_sum(float* out, const float* in,
    int batch, int HW, int C,
    cudaStream_t s)
{
    int total = batch * C;
    int block = 128;
    spatial_reduce_sum_kernel<<<grid_for(total, block), block, 0, s>>>(
        out, in, total, HW, C);
}

/* ===================== channel concat / split (NHWC) ===================== */

/* out[..., 0..Ca-1] = a[...,:], out[..., Ca..Ca+Cb-1] = b[...,:] */
__global__ void channel_cat_kernel(float* out, const float* a, const float* b,
    int spatial, int Ca, int Cb)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int Co = Ca + Cb;
    int total = spatial * Co;
    for (; idx < total; idx += st) {
        int c = idx % Co;
        int hw = idx / Co;
        out[idx] = (c < Ca) ? a[hw * Ca + c] : b[hw * Cb + (c - Ca)];
    }
}

__global__ void channel_split_kernel(float* grad_a, float* grad_b,
    const float* grad_out,
    int spatial, int Ca, int Cb)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int Co = Ca + Cb;
    int total = spatial * Co;
    for (; idx < total; idx += st) {
        int c = idx % Co;
        int hw = idx / Co;
        if (c < Ca)
            grad_a[hw * Ca + c] = grad_out[idx];
        else
            grad_b[hw * Cb + (c - Ca)] = grad_out[idx];
    }
}

__global__ void weighted_channel_split_kernel(float* grad_a, float* grad_b,
    const float* grad_out,
    float wa, float wb,
    int spatial, int Ca, int Cb)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int st = gridDim.x * blockDim.x;
    int Co = Ca + Cb;
    int total = spatial * Co;
    for (; idx < total; idx += st) {
        int c = idx % Co;
        int hw = idx / Co;
        if (c < Ca)
            grad_a[hw * Ca + c] = wa * grad_out[idx];
        else
            grad_b[hw * Cb + (c - Ca)] = wb * grad_out[idx];
    }
}

extern "C" void launch_channel_cat(float* out, const float* a, const float* b,
    int spatial, int Ca, int Cb, cudaStream_t s)
{
    int block = 256;
    channel_cat_kernel<<<grid_for(spatial * (Ca + Cb), block), block, 0, s>>>(
        out, a, b, spatial, Ca, Cb);
}

extern "C" void launch_channel_split(float* grad_a, float* grad_b,
    const float* grad_out,
    int spatial, int Ca, int Cb, cudaStream_t s)
{
    int block = 256;
    channel_split_kernel<<<grid_for(spatial * (Ca + Cb), block), block, 0, s>>>(
        grad_a, grad_b, grad_out, spatial, Ca, Cb);
}

static void mp_cat_weights(float t, int Na, int Nb, float* wa, float* wb)
{
    float denom = (1.0f - t) * (1.0f - t) + t * t;
    float C = sqrtf((float)(Na + Nb) / denom);
    *wa = C / sqrtf((float)Na) * (1.0f - t);
    *wb = C / sqrtf((float)Nb) * t;
}

extern "C" void launch_mp_channel_cat(float* out, const float* a, const float* b,
    float t, int spatial, int Ca, int Cb, cudaStream_t s)
{
    float wa, wb;
    mp_cat_weights(t, Ca, Cb, &wa, &wb);
    int block = 256;
    mp_cat_kernel<<<grid_for(spatial * (Ca + Cb), block), block, 0, s>>>(
        out, a, b, wa, wb, spatial, Ca, Cb);
}

extern "C" void launch_mp_channel_split(float* grad_a, float* grad_b,
    const float* grad_out,
    float t, int spatial, int Ca, int Cb, cudaStream_t s)
{
    float wa, wb;
    mp_cat_weights(t, Ca, Cb, &wa, &wb);
    int block = 256;
    weighted_channel_split_kernel<<<grid_for(spatial * (Ca + Cb), block), block, 0, s>>>(
        grad_a, grad_b, grad_out, wa, wb, spatial, Ca, Cb);
}

/* ===================== MSE loss ===================== */

/* Fused MSE forward + backward. One block per sample, threads cooperate over dim.
   loss_row[b] = mean_i (pred - target)^2,  grad = 2*(pred - target) / (batch*dim) */
__global__ void mse_fwd_bwd_kernel(float* grad, float* loss_row,
    const float* pred, const float* target,
    int batch, int dim)
{
    int b = blockIdx.x;
    if (b >= batch)
        return;
    const float* p = pred + (size_t)b * dim;
    const float* t = target + (size_t)b * dim;
    float* g = grad + (size_t)b * dim;
    float inv = 2.0f / (float)(batch * dim);
    float partial = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = p[i] - t[i];
        partial += d * d;
        g[i] = d * inv;
    }
    float total = block_reduce_sum(partial);
    if (threadIdx.x == 0)
        loss_row[b] = total / (float)dim;
}

extern "C" void launch_mse_fwd_bwd(float* grad, float* loss_row,
    const float* pred, const float* target,
    int batch, int dim, cudaStream_t s)
{
    int block = 256;
    mse_fwd_bwd_kernel<<<batch, block, 0, s>>>(
        grad, loss_row, pred, target, batch, dim);
}

/* ======================== clamp ======================== */

__global__ void clamp_kernel(float* x, float lo, float hi, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = fminf(fmaxf(x[i], lo), hi);
}

extern "C" void launch_clamp(float* x, float lo, float hi, int n, cudaStream_t s)
{
    int block = 256;
    clamp_kernel<<<(n + block - 1) / block, block, 0, s>>>(x, lo, hi, n);
}
