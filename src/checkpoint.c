#include "nn.h"
#include <stdlib.h>
#include <string.h>

#define CKPT_MAGIC 0x434B5054 /* "CKPT" */

static int rd(void* dst, size_t size, size_t n, FILE* f)
{
    return fread(dst, size, n, f) == n ? 0 : -1;
}

int checkpoint_save(const char* path, const ParamList* pl)
{
    FILE* f = fopen(path, "wb");
    if (!f)
        return -1;

    unsigned int magic = CKPT_MAGIC;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&pl->count, sizeof(pl->count), 1, f);
    for (int i = 0; i < pl->count; ++i)
        fwrite(&pl->items[i].size, sizeof(int), 1, f);

    for (int i = 0; i < pl->count; ++i) {
        size_t bytes = (size_t)pl->items[i].size * sizeof(float);
        float* buf = (float*)malloc(bytes);
        NN_CUDA_CHECK(cudaMemcpy(buf, pl->items[i].param, bytes,
            cudaMemcpyDeviceToHost));
        fwrite(buf, sizeof(float), pl->items[i].size, f);
        free(buf);
    }
    fclose(f);
    return 0;
}

int checkpoint_load(const char* path, ParamList* pl)
{
    FILE* f = fopen(path, "rb");
    if (!f)
        return -1;

    unsigned int magic;
    if (rd(&magic, sizeof(magic), 1, f) || magic != CKPT_MAGIC) {
        fclose(f);
        return -2;
    }

    int count;
    if (rd(&count, sizeof(count), 1, f)) {
        fclose(f);
        return -2;
    }
    if (count != pl->count) {
        fprintf(stderr, "checkpoint: param count mismatch (file=%d, model=%d)\n",
            count, pl->count);
        fclose(f);
        return -3;
    }

    for (int i = 0; i < count; ++i) {
        int size;
        if (rd(&size, sizeof(int), 1, f)) {
            fclose(f);
            return -2;
        }
        if (size != pl->items[i].size) {
            fprintf(stderr, "checkpoint: param %d size mismatch (file=%d, model=%d)\n",
                i, size, pl->items[i].size);
            fclose(f);
            return -4;
        }
    }

    for (int i = 0; i < count; ++i) {
        size_t bytes = (size_t)pl->items[i].size * sizeof(float);
        float* buf = (float*)malloc(bytes);
        if (rd(buf, sizeof(float), pl->items[i].size, f)) {
            free(buf);
            fclose(f);
            return -2;
        }
        NN_CUDA_CHECK(cudaMemcpy(pl->items[i].param, buf, bytes,
            cudaMemcpyHostToDevice));
        free(buf);
    }
    fclose(f);
    return 0;
}
