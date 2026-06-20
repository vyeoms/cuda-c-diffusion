#ifndef MNIST_IO_H
#define MNIST_IO_H

#include <stdio.h>
#include <stdlib.h>

static inline float* mnist_load_images(const char* path, int* n_out)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = (int)(sz / (28 * 28 * sizeof(float)));
    float* data = (float*)malloc(sz);
    if (fread(data, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "short read on %s\n", path);
        exit(1);
    }
    fclose(f);
    *n_out = n;
    return data;
}

static inline float* mnist_load_labels(const char* path, int* n_out)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = (int)(sz / sizeof(float));
    float* data = (float*)malloc(sz);
    if (fread(data, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "short read on %s\n", path);
        exit(1);
    }
    fclose(f);
    *n_out = n;
    return data;
}

#endif /* MNIST_IO_H */
