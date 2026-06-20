#ifndef GEMM_H
#define GEMM_H

#include "nn.h" /* cuBLAS handle + NN_CUBLAS_CHECK */

/* Row-major cuBLAS GEMM helpers; alpha scales the product, beta = 0.
   cuBLAS is column-major: the leading dimension is each matrix's row-major
   width, and the operands/ops follow the standard transpose idiom. */

/* Y[batch,out] = alpha * X[batch,in] @ W[in,out] */
static inline void gemm_forward(cublasHandle_t h, float alpha,
    const float* X, const float* W, float* Y,
    int batch, int in, int out)
{
    const float beta = 0.0f;
    NN_CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, out, batch, in,
        &alpha, W, out, X, in, &beta, Y, out));
}

/* gW[in,out] = alpha * X[batch,in]^T @ dZ[batch,out] */
static inline void gemm_dW(cublasHandle_t h, float alpha,
    const float* X, const float* dZ, float* gW,
    int batch, int in, int out)
{
    const float beta = 0.0f;
    NN_CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, out, in, batch,
        &alpha, dZ, out, X, in, &beta, gW, out));
}

/* dX[batch,in] = alpha * dZ[batch,out] @ W[in,out]^T */
static inline void gemm_dX(cublasHandle_t h, float alpha,
    const float* W, const float* dZ, float* dX,
    int batch, int in, int out)
{
    const float beta = 0.0f;
    NN_CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, in, batch, out,
        &alpha, W, out, dZ, out, &beta, dX, in));
}

/* dX += alpha * dZ @ W^T  (accumulate with beta=1 for attention gradient fan-in) */
static inline void gemm_dX_beta(cublasHandle_t h, float alpha, float beta,
    const float* W, const float* dZ, float* dX,
    int batch, int in, int out)
{
    NN_CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, in, batch, out,
        &alpha, W, out, dZ, out, &beta, dX, in));
}

/* ================ batched GEMM (row-major, strided) ================ */

/* C[i] = alpha * A[i] @ B[i],  each [M,K] @ [K,N] = [M,N] */
static inline void gemm_batched_nn(cublasHandle_t h, float alpha,
    const float* A, const float* B, float* C,
    int M, int K, int N, int batch)
{
    const float beta = 0.0f;
    NN_CUBLAS_CHECK(cublasSgemmStridedBatched(h,
        CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha, B, N, (long long)K * N,
        A, K, (long long)M * K,
        &beta, C, N, (long long)M * N, batch));
}

/* C[i] = alpha * A[i] @ B[i]^T,  each [M,K] @ [N,K]^T = [M,N] */
static inline void gemm_batched_nt(cublasHandle_t h, float alpha,
    const float* A, const float* B, float* C,
    int M, int K, int N, int batch)
{
    const float beta = 0.0f;
    NN_CUBLAS_CHECK(cublasSgemmStridedBatched(h,
        CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        &alpha, B, K, (long long)N * K,
        A, K, (long long)M * K,
        &beta, C, N, (long long)M * N, batch));
}

/* C[i] = alpha * A[i]^T @ B[i],  each [K,M]^T @ [K,N] = [M,N] */
static inline void gemm_batched_tn(cublasHandle_t h, float alpha,
    const float* A, const float* B, float* C,
    int M, int K, int N, int batch)
{
    const float beta = 0.0f;
    NN_CUBLAS_CHECK(cublasSgemmStridedBatched(h,
        CUBLAS_OP_N, CUBLAS_OP_T, N, M, K,
        &alpha, B, N, (long long)K * N,
        A, M, (long long)K * M,
        &beta, C, N, (long long)M * N, batch));
}

#endif /* GEMM_H */
