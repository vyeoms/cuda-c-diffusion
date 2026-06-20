#include "nn.h"

struct Loss {
    int C, max_batch;
    float* d_loss_row; /* per-sample loss (device) */
};

Loss* loss_create(int max_batch)
{
    Loss* l = (Loss*)malloc(sizeof(Loss));
    l->C = 0;
    l->max_batch = max_batch;
    l->d_loss_row = nn_device_alloc((size_t)max_batch);
    return l;
}

Loss* loss_softmax_xent_create(int num_classes, int max_batch)
{
    Loss* l = loss_create(max_batch);
    l->C = num_classes;
    return l;
}

float loss_forward_backward(Loss* l, const float* logits, const int* targets,
    float* grad_logits, int batch, Context* ctx)
{
    launch_softmax_xent(grad_logits, l->d_loss_row, logits, targets,
        batch, l->C, ctx->stream);

    float* h = (float*)malloc(batch * sizeof(float));
    NN_CUDA_CHECK(cudaMemcpy(h, l->d_loss_row, batch * sizeof(float),
        cudaMemcpyDeviceToHost));
    float total = 0.0f;
    for (int i = 0; i < batch; ++i)
        total += h[i];
    free(h);
    return total / (float)batch;
}

float mse_forward_backward(Loss* l, const float* pred, const float* target,
    float* grad, int batch, int dim, Context* ctx)
{
    launch_mse_fwd_bwd(grad, l->d_loss_row, pred, target, batch, dim, ctx->stream);

    float* h = (float*)malloc(batch * sizeof(float));
    NN_CUDA_CHECK(cudaMemcpy(h, l->d_loss_row, batch * sizeof(float),
        cudaMemcpyDeviceToHost));
    float total = 0.0f;
    for (int i = 0; i < batch; ++i)
        total += h[i];
    free(h);
    return total / (float)batch;
}

float mse_weighted_forward_backward(Loss* l, const float* pred, const float* target,
    float* grad, int batch, int dim,
    const float* sample_weights, Context* ctx)
{
    launch_mse_fwd_bwd(grad, l->d_loss_row, pred, target, batch, dim, ctx->stream);

    float* h = (float*)malloc(batch * sizeof(float));
    NN_CUDA_CHECK(cudaMemcpy(h, l->d_loss_row, batch * sizeof(float),
        cudaMemcpyDeviceToHost));
    float total = 0.0f;
    for (int i = 0; i < batch; ++i) {
        total += h[i] * sample_weights[i];
        launch_weighted_sum(grad + (size_t)i * dim, grad + (size_t)i * dim,
            grad + (size_t)i * dim,
            sample_weights[i], 0.0f, dim, ctx->stream);
    }
    free(h);
    return total / (float)batch;
}

void loss_free(Loss* l)
{
    cudaFree(l->d_loss_row);
    free(l);
}
