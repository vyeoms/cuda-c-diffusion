#include "nn.h"
#include <stdio.h>
#include <stdlib.h>

static float accuracy(const float* d_logits, const int* h_targets, int batch, int C)
{
    float* h = (float*)malloc((size_t)batch * C * sizeof(float));
    NN_CUDA_CHECK(cudaMemcpy(h, d_logits, (size_t)batch * C * sizeof(float),
        cudaMemcpyDeviceToHost));
    int correct = 0;
    for (int r = 0; r < batch; ++r) {
        int best = 0;
        for (int j = 1; j < C; ++j)
            if (h[r * C + j] > h[r * C + best])
                best = j;
        if (best == h_targets[r])
            ++correct;
    }
    free(h);
    return (float)correct / (float)batch;
}

int main(void)
{
    int batch = 8;
    int in_dim = 32;
    int hidden = 64;
    int classes = 10;
    int cemb = 32;
    int steps = 2000;

    Context ctx;
    NN_CUBLAS_CHECK(cublasCreate(&ctx.cublas));
    NN_CUBLAS_CHECK(cublasSetMathMode(ctx.cublas, CUBLAS_TF32_TENSOR_OP_MATH));
    ctx.stream = 0;

    NN_CURAND_CHECK(curandCreateGenerator(&ctx.curand, CURAND_RNG_PSEUDO_DEFAULT));
    NN_CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx.curand, 0ULL));
    NN_CURAND_CHECK(curandSetStream(ctx.curand, ctx.stream));

    ctx.n_cond = 0;
    int cond_slot = ctx_add_cond(&ctx, cemb, batch);

    /* ---- embedding pipeline: noise scalar -> Fourier features -> linear -> mp_silu ---- */
    Module* emb_mods[] = {
        mp_fourier_create(&ctx, cemb, 1.0f),
        mp_linear_create(&ctx, cemb, cemb, 1.0f),
        activation_create(cemb, ACT_MP_SILU),
    };
    Sequential* embed = sequential_create(emb_mods, 3, batch);

    /* ---- main conditioned resnet ---- */
    Module* mods[] = {
        mp_linear_create(&ctx, in_dim, hidden, 1.0f), /* stem */
        cond_residual_create(&ctx, hidden, cemb, batch, 0.3f, cond_slot),
        cond_residual_create(&ctx, hidden, cemb, batch, 0.3f, cond_slot),
        mp_linear_create(&ctx, hidden, classes, 1.0f), /* head -> logits */
    };
    Sequential* model = sequential_create(mods, 4, batch);

    ParamList params;
    param_list_init(&params);
    sequential_parameters(embed, &params);
    sequential_parameters(model, &params);
    Optimizer* opt = optimizer_adam(&params, 1e-2f, 0.9f, 0.999f, 1e-8f);
    Loss* loss = loss_softmax_xent_create(classes, batch);

    /* inputs ~ N(0,1) so MP holds from layer 1 */
    float* d_x = nn_device_alloc((size_t)batch * in_dim);
    nn_fill_normal(ctx.curand, d_x, (size_t)batch * in_dim, 0.0f, 1.0f);
    float* d_noise = nn_device_alloc((size_t)batch);
    nn_fill_normal(ctx.curand, d_noise, (size_t)batch, 0.0f, 1.0f);

    float* d_u = nn_device_alloc((size_t)batch);
    NN_CURAND_CHECK(curandGenerateUniform(ctx.curand, d_u, batch));
    float* h_u = (float*)malloc(batch * sizeof(float));
    NN_CUDA_CHECK(cudaMemcpy(h_u, d_u, batch * sizeof(float), cudaMemcpyDeviceToHost));
    int* h_t = (int*)malloc(batch * sizeof(int));
    for (int i = 0; i < batch; ++i)
        h_t[i] = (int)(h_u[i] * classes);
    free(h_u);
    cudaFree(d_u);
    int* d_t;
    NN_CUDA_CHECK(cudaMalloc((void**)&d_t, batch * sizeof(int)));
    NN_CUDA_CHECK(cudaMemcpy(d_t, h_t, batch * sizeof(int), cudaMemcpyHostToDevice));

    float* grad_logits = nn_device_alloc((size_t)batch * classes);

    /* ---- training loop ---- */
    for (int step = 1; step <= steps; ++step) {
        ctx_set_cond(&ctx, cond_slot, NULL);
        sequential_forward(embed, d_noise, batch, &ctx);
        ctx_set_cond(&ctx, cond_slot, sequential_output(embed));
        sequential_forward(model, d_x, batch, &ctx);

        float l = loss_forward_backward(loss, sequential_output(model), d_t,
            grad_logits, batch, &ctx);

        ctx_zero_cond_grad(&ctx, batch);
        sequential_backward(model, grad_logits, batch, &ctx);
        sequential_backward(embed, ctx.cond[cond_slot].grad, batch, &ctx);
        optimizer_step(opt, &params, &ctx);

        if (step == 1 || step % 200 == 0) {
            ctx_set_cond(&ctx, cond_slot, NULL);
            sequential_forward(embed, d_noise, batch, &ctx);
            ctx_set_cond(&ctx, cond_slot, sequential_output(embed));
            sequential_forward(model, d_x, batch, &ctx);
            float acc = accuracy(sequential_output(model), h_t, batch, classes);
            printf("step %4d   loss %.4f   acc %.0f%%\n", step, l, acc * 100.0f);
        }
    }

    free(h_t);
    cudaFree(d_x);
    cudaFree(d_noise);
    cudaFree(d_t);
    cudaFree(grad_logits);
    param_list_free(&params);
    ctx_free_cond(&ctx);
    loss_free(loss);
    optimizer_free(opt);
    sequential_free(model);
    sequential_free(embed);
    curandDestroyGenerator(ctx.curand);
    cublasDestroy(ctx.cublas);
    return 0;
}
