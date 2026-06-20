#include "gemm.h"
#include "nn.h"
#include <math.h>
#include <stdlib.h>

typedef struct {
    int C, heads, head_dim, S, H, W, max_batch, mp_mode;
    float eps, scale;

    float *Wq, *gWq, *Wk, *gWk, *Wv, *gWv, *Wo, *gWo;
    float *bq, *gbq, *bk, *gbk, *bv, *gbv, *bo, *gbo;
    float *gain_q, *g_gain_q, *gain_k, *g_gain_k;
    float *gain_v, *g_gain_v, *gain_o, *g_gain_o;

    float *q_perm, *k_perm, *v_perm;
    float *a_buf, *attn_out_flat;
    float *scratch1, *grad_scores;
} AttentionState;

static void attn_project_forward(AttentionState* st, const float* X,
    float* W, float* buf_perm, int BS,
    float* gain, Context* ctx)
{
    if (st->mp_mode) {
        launch_weight_normalize(W, st->C, st->C, st->eps, ctx->stream);
        gemm_forward(ctx->cublas, 1.0f, X, W, st->scratch1, BS, st->C, st->C);
        launch_scale_by_scalar(st->scratch1, gain, BS * st->C, ctx->stream);
    } else {
        gemm_forward(ctx->cublas, 1.0f, X, W, st->scratch1, BS, st->C, st->C);
    }
    launch_permute_0213(buf_perm, st->scratch1, BS / st->S, st->S,
        st->heads, st->head_dim, ctx->stream);
}

static void attn_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    AttentionState* st = (AttentionState*)self->state;
    int BS = batch * st->S;
    int BH = batch * st->heads;

    if (!st->mp_mode) {
        gemm_forward(ctx->cublas, 1.0f, in, st->Wq, st->scratch1, BS, st->C, st->C);
        launch_add_bias(st->scratch1, st->bq, BS, st->C, ctx->stream);
        launch_permute_0213(st->q_perm, st->scratch1, batch, st->S, st->heads, st->head_dim, ctx->stream);

        gemm_forward(ctx->cublas, 1.0f, in, st->Wk, st->scratch1, BS, st->C, st->C);
        launch_add_bias(st->scratch1, st->bk, BS, st->C, ctx->stream);
        launch_permute_0213(st->k_perm, st->scratch1, batch, st->S, st->heads, st->head_dim, ctx->stream);

        gemm_forward(ctx->cublas, 1.0f, in, st->Wv, st->scratch1, BS, st->C, st->C);
        launch_add_bias(st->scratch1, st->bv, BS, st->C, ctx->stream);
        launch_permute_0213(st->v_perm, st->scratch1, batch, st->S, st->heads, st->head_dim, ctx->stream);
    } else {
        attn_project_forward(st, in, st->Wq, st->q_perm, BS, st->gain_q, ctx);
        attn_project_forward(st, in, st->Wk, st->k_perm, BS, st->gain_k, ctx);
        attn_project_forward(st, in, st->Wv, st->v_perm, BS, st->gain_v, ctx);
    }

    gemm_batched_nt(ctx->cublas, st->scale,
        st->q_perm, st->k_perm, st->a_buf,
        st->S, st->head_dim, st->S, BH);
    launch_attn_softmax(st->a_buf, st->a_buf, BH * st->S, st->S, ctx->stream);

    gemm_batched_nn(ctx->cublas, 1.0f,
        st->a_buf, st->v_perm, st->scratch1,
        st->S, st->S, st->head_dim, BH);
    launch_permute_0213(st->attn_out_flat, st->scratch1, batch, st->heads,
        st->S, st->head_dim, ctx->stream);

    if (st->mp_mode) {
        launch_weight_normalize(st->Wo, st->C, st->C, st->eps, ctx->stream);
        gemm_forward(ctx->cublas, 1.0f, st->attn_out_flat, st->Wo, out,
            BS, st->C, st->C);
        launch_scale_by_scalar(out, st->gain_o, BS * st->C, ctx->stream);
    } else {
        gemm_forward(ctx->cublas, 1.0f, st->attn_out_flat, st->Wo, out,
            BS, st->C, st->C);
        launch_add_bias(out, st->bo, BS, st->C, ctx->stream);
    }
}

static void attn_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)out;
    AttentionState* st = (AttentionState*)self->state;
    int BS = batch * st->S;
    int BH = batch * st->heads;

    /* --- output projection backward --- */
    gemm_dW(ctx->cublas, 1.0f, st->attn_out_flat, grad_out, st->gWo,
        BS, st->C, st->C);
    if (st->mp_mode) {
        NN_CUDA_CHECK(cudaMemsetAsync(st->g_gain_o, 0, sizeof(float), ctx->stream));
        launch_weight_normalize_backward(st->gWo, st->Wo, st->gain_o, st->g_gain_o,
            st->C, st->C, ctx->stream);
        gemm_dX(ctx->cublas, 1.0f, st->Wo, grad_out, st->scratch1, BS, st->C, st->C);
        launch_scale_by_scalar(st->scratch1, st->gain_o, BS * st->C, ctx->stream);
    } else {
        launch_bias_grad(st->gbo, grad_out, BS, st->C, ctx->stream);
        gemm_dX(ctx->cublas, 1.0f, st->Wo, grad_out, st->scratch1, BS, st->C, st->C);
    }

    /* scratch1 = grad_attn_out_flat [BS, C] */
    /* permute to [B, heads, S, D] -> reuse attn_out_flat buffer */
    launch_permute_0213(st->attn_out_flat, st->scratch1, batch, st->S,
        st->heads, st->head_dim, ctx->stream);
    /* attn_out_flat now holds grad_attn_out_perm [BH, S, D] */
    float* grad_attn_perm = st->attn_out_flat;

    /* --- attention backward --- */
    gemm_batched_tn(ctx->cublas, 1.0f,
        st->a_buf, grad_attn_perm, st->scratch1,
        st->S, st->S, st->head_dim, BH);
    /* scratch1 = grad_V_perm [BH, S, D] */

    gemm_batched_nt(ctx->cublas, 1.0f,
        grad_attn_perm, st->v_perm, st->grad_scores,
        st->S, st->head_dim, st->S, BH);
    /* grad_scores = grad_A [BH, S, S] */

    launch_attn_softmax_backward(st->grad_scores, st->grad_scores, st->a_buf,
        BH * st->S, st->S, ctx->stream);
    /* grad_scores now has softmax backward applied, in-place */

    /* inverse permute grad_V_perm -> v_perm buffer (reuse) */
    launch_permute_0213(st->v_perm, st->scratch1, batch, st->heads,
        st->S, st->head_dim, ctx->stream);
    /* v_perm now holds grad_V_flat [BS, C] */

    gemm_batched_nn(ctx->cublas, st->scale,
        st->grad_scores, st->k_perm, st->scratch1,
        st->S, st->S, st->head_dim, BH);
    /* scratch1 = grad_Q_perm */

    gemm_batched_tn(ctx->cublas, st->scale,
        st->grad_scores, st->q_perm, grad_attn_perm,
        st->S, st->S, st->head_dim, BH);
    /* grad_attn_perm = grad_K_perm (reusing attn_out_flat) */

    /* inverse permute grad_Q, grad_K */
    launch_permute_0213(st->q_perm, st->scratch1, batch, st->heads,
        st->S, st->head_dim, ctx->stream);
    /* q_perm = grad_Q_flat */
    launch_permute_0213(st->k_perm, grad_attn_perm, batch, st->heads,
        st->S, st->head_dim, ctx->stream);
    /* k_perm = grad_K_flat */

    /* --- Q, K, V projection backward --- */
    float *grad_Q = st->q_perm, *grad_K = st->k_perm, *grad_V = st->v_perm;

    if (st->mp_mode) {
        /* Q */
        gemm_dW(ctx->cublas, 1.0f, in, grad_Q, st->gWq, BS, st->C, st->C);
        NN_CUDA_CHECK(cudaMemsetAsync(st->g_gain_q, 0, sizeof(float), ctx->stream));
        launch_weight_normalize_backward(st->gWq, st->Wq, st->gain_q, st->g_gain_q,
            st->C, st->C, ctx->stream);
        gemm_dX(ctx->cublas, 1.0f, st->Wq, grad_Q, grad_in, BS, st->C, st->C);
        launch_scale_by_scalar(grad_in, st->gain_q, BS * st->C, ctx->stream);

        /* K (accumulate) */
        gemm_dW(ctx->cublas, 1.0f, in, grad_K, st->gWk, BS, st->C, st->C);
        NN_CUDA_CHECK(cudaMemsetAsync(st->g_gain_k, 0, sizeof(float), ctx->stream));
        launch_weight_normalize_backward(st->gWk, st->Wk, st->gain_k, st->g_gain_k,
            st->C, st->C, ctx->stream);
        gemm_dX(ctx->cublas, 1.0f, st->Wk, grad_K, st->scratch1, BS, st->C, st->C);
        launch_scale_by_scalar(st->scratch1, st->gain_k, BS * st->C, ctx->stream);
        launch_weighted_sum(grad_in, grad_in, st->scratch1, 1.0f, 1.0f,
            BS * st->C, ctx->stream);

        /* V (accumulate) */
        gemm_dW(ctx->cublas, 1.0f, in, grad_V, st->gWv, BS, st->C, st->C);
        NN_CUDA_CHECK(cudaMemsetAsync(st->g_gain_v, 0, sizeof(float), ctx->stream));
        launch_weight_normalize_backward(st->gWv, st->Wv, st->gain_v, st->g_gain_v,
            st->C, st->C, ctx->stream);
        gemm_dX(ctx->cublas, 1.0f, st->Wv, grad_V, st->scratch1, BS, st->C, st->C);
        launch_scale_by_scalar(st->scratch1, st->gain_v, BS * st->C, ctx->stream);
        launch_weighted_sum(grad_in, grad_in, st->scratch1, 1.0f, 1.0f,
            BS * st->C, ctx->stream);
    } else {
        gemm_dW(ctx->cublas, 1.0f, in, grad_Q, st->gWq, BS, st->C, st->C);
        launch_bias_grad(st->gbq, grad_Q, BS, st->C, ctx->stream);
        gemm_dX(ctx->cublas, 1.0f, st->Wq, grad_Q, grad_in, BS, st->C, st->C);

        gemm_dW(ctx->cublas, 1.0f, in, grad_K, st->gWk, BS, st->C, st->C);
        launch_bias_grad(st->gbk, grad_K, BS, st->C, ctx->stream);
        gemm_dX_beta(ctx->cublas, 1.0f, 1.0f, st->Wk, grad_K, grad_in,
            BS, st->C, st->C);

        gemm_dW(ctx->cublas, 1.0f, in, grad_V, st->gWv, BS, st->C, st->C);
        launch_bias_grad(st->gbv, grad_V, BS, st->C, ctx->stream);
        gemm_dX_beta(ctx->cublas, 1.0f, 1.0f, st->Wv, grad_V, grad_in,
            BS, st->C, st->C);
    }
}

static void attn_parameters(Module* self, ParamList* pl)
{
    AttentionState* st = (AttentionState*)self->state;
    int CC = st->C * st->C;
    param_list_add(pl, st->Wq, st->gWq, CC);
    param_list_add(pl, st->Wk, st->gWk, CC);
    param_list_add(pl, st->Wv, st->gWv, CC);
    param_list_add(pl, st->Wo, st->gWo, CC);
    if (st->mp_mode) {
        param_list_add(pl, st->gain_q, st->g_gain_q, 1);
        param_list_add(pl, st->gain_k, st->g_gain_k, 1);
        param_list_add(pl, st->gain_v, st->g_gain_v, 1);
        param_list_add(pl, st->gain_o, st->g_gain_o, 1);
    } else {
        param_list_add(pl, st->bq, st->gbq, st->C);
        param_list_add(pl, st->bk, st->gbk, st->C);
        param_list_add(pl, st->bv, st->gbv, st->C);
        param_list_add(pl, st->bo, st->gbo, st->C);
    }
}

static void attn_free(Module* self)
{
    AttentionState* st = (AttentionState*)self->state;
    cudaFree(st->Wq);
    cudaFree(st->gWq);
    cudaFree(st->Wk);
    cudaFree(st->gWk);
    cudaFree(st->Wv);
    cudaFree(st->gWv);
    cudaFree(st->Wo);
    cudaFree(st->gWo);
    if (st->mp_mode) {
        cudaFree(st->gain_q);
        cudaFree(st->g_gain_q);
        cudaFree(st->gain_k);
        cudaFree(st->g_gain_k);
        cudaFree(st->gain_v);
        cudaFree(st->g_gain_v);
        cudaFree(st->gain_o);
        cudaFree(st->g_gain_o);
    } else {
        cudaFree(st->bq);
        cudaFree(st->gbq);
        cudaFree(st->bk);
        cudaFree(st->gbk);
        cudaFree(st->bv);
        cudaFree(st->gbv);
        cudaFree(st->bo);
        cudaFree(st->gbo);
    }
    cudaFree(st->q_perm);
    cudaFree(st->k_perm);
    cudaFree(st->v_perm);
    cudaFree(st->a_buf);
    cudaFree(st->attn_out_flat);
    cudaFree(st->scratch1);
    cudaFree(st->grad_scores);
    free(st);
    free(self);
}

Module* attention_create(Context* ctx, int C, int heads, int H, int W,
    int max_batch, int mp_mode, float gain)
{
    Module* m = (Module*)malloc(sizeof(Module));
    AttentionState* st = (AttentionState*)malloc(sizeof(AttentionState));

    st->C = C;
    st->heads = heads;
    st->head_dim = C / heads;
    st->S = H * W;
    st->H = H;
    st->W = W;
    st->max_batch = max_batch;
    st->mp_mode = mp_mode;
    st->eps = 1e-4f;
    st->scale = 1.0f / sqrtf((float)st->head_dim);

    size_t CC = (size_t)C * C;
    st->Wq = nn_device_alloc(CC);
    st->gWq = nn_device_alloc(CC);
    st->Wk = nn_device_alloc(CC);
    st->gWk = nn_device_alloc(CC);
    st->Wv = nn_device_alloc(CC);
    st->gWv = nn_device_alloc(CC);
    st->Wo = nn_device_alloc(CC);
    st->gWo = nn_device_alloc(CC);

    if (mp_mode) {
        st->bq = NULL;
        st->gbq = NULL;
        st->bk = NULL;
        st->gbk = NULL;
        st->bv = NULL;
        st->gbv = NULL;
        st->bo = NULL;
        st->gbo = NULL;
        st->gain_q = nn_device_alloc(1);
        st->g_gain_q = nn_device_alloc(1);
        st->gain_k = nn_device_alloc(1);
        st->g_gain_k = nn_device_alloc(1);
        st->gain_v = nn_device_alloc(1);
        st->g_gain_v = nn_device_alloc(1);
        st->gain_o = nn_device_alloc(1);
        st->g_gain_o = nn_device_alloc(1);
        nn_fill_normal(ctx->curand, st->Wq, CC, 0.0f, 1.0f);
        nn_fill_normal(ctx->curand, st->Wk, CC, 0.0f, 1.0f);
        nn_fill_normal(ctx->curand, st->Wv, CC, 0.0f, 1.0f);
        nn_fill_normal(ctx->curand, st->Wo, CC, 0.0f, 1.0f);
        float g = gain;
        NN_CUDA_CHECK(cudaMemcpy(st->gain_q, &g, sizeof(float), cudaMemcpyHostToDevice));
        NN_CUDA_CHECK(cudaMemcpy(st->gain_k, &g, sizeof(float), cudaMemcpyHostToDevice));
        NN_CUDA_CHECK(cudaMemcpy(st->gain_v, &g, sizeof(float), cudaMemcpyHostToDevice));
        NN_CUDA_CHECK(cudaMemcpy(st->gain_o, &g, sizeof(float), cudaMemcpyHostToDevice));
    } else {
        st->gain_q = NULL;
        st->g_gain_q = NULL;
        st->gain_k = NULL;
        st->g_gain_k = NULL;
        st->gain_v = NULL;
        st->g_gain_v = NULL;
        st->gain_o = NULL;
        st->g_gain_o = NULL;
        float std = sqrtf(2.0f / (float)C);
        nn_fill_normal(ctx->curand, st->Wq, CC, 0.0f, std);
        nn_fill_normal(ctx->curand, st->Wk, CC, 0.0f, std);
        nn_fill_normal(ctx->curand, st->Wv, CC, 0.0f, std);
        nn_fill_normal(ctx->curand, st->Wo, CC, 0.0f, std);
        st->bq = nn_device_alloc(C);
        st->gbq = nn_device_alloc(C);
        st->bk = nn_device_alloc(C);
        st->gbk = nn_device_alloc(C);
        st->bv = nn_device_alloc(C);
        st->gbv = nn_device_alloc(C);
        st->bo = nn_device_alloc(C);
        st->gbo = nn_device_alloc(C);
        NN_CUDA_CHECK(cudaMemset(st->bq, 0, C * sizeof(float)));
        NN_CUDA_CHECK(cudaMemset(st->bk, 0, C * sizeof(float)));
        NN_CUDA_CHECK(cudaMemset(st->bv, 0, C * sizeof(float)));
        NN_CUDA_CHECK(cudaMemset(st->bo, 0, C * sizeof(float)));
    }

    size_t BSC = (size_t)max_batch * st->S * C;
    size_t BHSS = (size_t)max_batch * heads * st->S * st->S;
    st->q_perm = nn_device_alloc(BSC);
    st->k_perm = nn_device_alloc(BSC);
    st->v_perm = nn_device_alloc(BSC);
    st->a_buf = nn_device_alloc(BHSS);
    st->attn_out_flat = nn_device_alloc(BSC);
    st->scratch1 = nn_device_alloc(BSC);
    st->grad_scores = nn_device_alloc(BHSS);

    int dim = H * W * C;
    m->name = "attention";
    m->in_dim = dim;
    m->out_dim = dim;
    m->state = st;
    m->forward = attn_forward;
    m->backward = attn_backward;
    m->parameters = attn_parameters;
    m->free = attn_free;
    return m;
}
