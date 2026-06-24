#include "nn.h"
#include <stdlib.h>
#include <string.h>

#define CKPT_MAGIC 0x434B5054 /* "CKPT" : weights only */
#define CKPT_TRAIN_MAGIC 0x434B5452 /* "CKTR" : full training state */

static int rd(void* dst, size_t size, size_t n, FILE* f)
{
    return fread(dst, size, n, f) == n ? 0 : -1;
}

/* Serialize one ParamList: count, per-tensor sizes, then the float payloads
   (copied device->host). Shared by the weights-only and training checkpoints. */
static void write_list(FILE* f, const ParamList* pl)
{
    fwrite(&pl->count, sizeof(int), 1, f);
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
}

/* Inverse of write_list, validating count and per-tensor sizes against `pl`. */
static int read_list(FILE* f, ParamList* pl)
{
    int count;
    if (rd(&count, sizeof(int), 1, f))
        return -2;
    if (count != pl->count) {
        fprintf(stderr, "checkpoint: param count mismatch (file=%d, model=%d)\n",
            count, pl->count);
        return -3;
    }
    for (int i = 0; i < count; ++i) {
        int size;
        if (rd(&size, sizeof(int), 1, f))
            return -2;
        if (size != pl->items[i].size) {
            fprintf(stderr, "checkpoint: param %d size mismatch (file=%d, model=%d)\n",
                i, size, pl->items[i].size);
            return -4;
        }
    }
    for (int i = 0; i < count; ++i) {
        size_t bytes = (size_t)pl->items[i].size * sizeof(float);
        float* buf = (float*)malloc(bytes);
        if (rd(buf, sizeof(float), pl->items[i].size, f)) {
            free(buf);
            return -2;
        }
        NN_CUDA_CHECK(cudaMemcpy(pl->items[i].param, buf, bytes,
            cudaMemcpyHostToDevice));
        free(buf);
    }
    return 0;
}

int checkpoint_save(const char* path, const ParamList* pl)
{
    FILE* f = fopen(path, "wb");
    if (!f)
        return -1;
    unsigned int magic = CKPT_MAGIC;
    fwrite(&magic, sizeof(magic), 1, f);
    write_list(f, pl);
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
    int rc = read_list(f, pl);
    fclose(f);
    return rc;
}

int checkpoint_save_training(const char* path, int step, int adam_t,
    const ParamList* params, const ParamList* ema_shadow,
    const ParamList* adam_m, const ParamList* adam_v)
{
    FILE* f = fopen(path, "wb");
    if (!f)
        return -1;
    unsigned int magic = CKPT_TRAIN_MAGIC;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&step, sizeof(int), 1, f);
    fwrite(&adam_t, sizeof(int), 1, f);
    write_list(f, params);
    write_list(f, ema_shadow);
    write_list(f, adam_m);
    write_list(f, adam_v);
    fclose(f);
    return 0;
}

int checkpoint_load_training(const char* path, int* step, int* adam_t,
    ParamList* params, ParamList* ema_shadow,
    ParamList* adam_m, ParamList* adam_v)
{
    FILE* f = fopen(path, "rb");
    if (!f)
        return -1;
    unsigned int magic;
    if (rd(&magic, sizeof(magic), 1, f) || magic != CKPT_TRAIN_MAGIC) {
        fclose(f);
        return -2;
    }
    if (rd(step, sizeof(int), 1, f) || rd(adam_t, sizeof(int), 1, f)) {
        fclose(f);
        return -2;
    }
    int rc = read_list(f, params);
    if (!rc)
        rc = read_list(f, ema_shadow);
    if (!rc)
        rc = read_list(f, adam_m);
    if (!rc)
        rc = read_list(f, adam_v);
    fclose(f);
    return rc;
}
