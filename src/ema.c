#include "nn.h"
#include <stdlib.h>
#include <string.h>

struct EMA {
    float decay;
    int count;
    int* sizes;
    float** shadow;
};

EMA* ema_create(const ParamList* pl, float decay)
{
    EMA* ema = (EMA*)malloc(sizeof(EMA));
    ema->decay = decay;
    ema->count = pl->count;
    ema->sizes = (int*)malloc(pl->count * sizeof(int));
    ema->shadow = (float**)malloc(pl->count * sizeof(float*));
    for (int i = 0; i < pl->count; ++i) {
        ema->sizes[i] = pl->items[i].size;
        ema->shadow[i] = nn_device_alloc((size_t)pl->items[i].size);
        NN_CUDA_CHECK(cudaMemcpy(ema->shadow[i], pl->items[i].param,
            pl->items[i].size * sizeof(float),
            cudaMemcpyDeviceToDevice));
    }
    return ema;
}

void ema_update(EMA* ema, const ParamList* pl, Context* ctx)
{
    for (int i = 0; i < ema->count; ++i)
        launch_weighted_sum(ema->shadow[i], ema->shadow[i], pl->items[i].param,
            ema->decay, 1.0f - ema->decay,
            ema->sizes[i], ctx->stream);
}

void ema_swap(EMA* ema, ParamList* pl, Context* ctx)
{
    int max_size = 0;
    for (int i = 0; i < ema->count; ++i)
        if (ema->sizes[i] > max_size)
            max_size = ema->sizes[i];
    float* tmp = nn_device_alloc((size_t)max_size);
    for (int i = 0; i < ema->count; ++i) {
        size_t bytes = (size_t)ema->sizes[i] * sizeof(float);
        NN_CUDA_CHECK(cudaMemcpyAsync(tmp, pl->items[i].param, bytes,
            cudaMemcpyDeviceToDevice, ctx->stream));
        NN_CUDA_CHECK(cudaMemcpyAsync(pl->items[i].param, ema->shadow[i], bytes,
            cudaMemcpyDeviceToDevice, ctx->stream));
        NN_CUDA_CHECK(cudaMemcpyAsync(ema->shadow[i], tmp, bytes,
            cudaMemcpyDeviceToDevice, ctx->stream));
    }
    cudaFree(tmp);
}

void ema_shadow_list(const EMA* ema, ParamList* shadow)
{
    for (int i = 0; i < ema->count; ++i)
        param_list_add(shadow, ema->shadow[i], NULL, ema->sizes[i]);
}

void ema_free(EMA* ema)
{
    for (int i = 0; i < ema->count; ++i)
        cudaFree(ema->shadow[i]);
    free(ema->shadow);
    free(ema->sizes);
    free(ema);
}
