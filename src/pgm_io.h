#ifndef PGM_IO_H
#define PGM_IO_H

#include <stdio.h>
#include <stdlib.h>

static inline void save_pgm(const char* path, const float* pixels, int H, int W)
{
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        return;
    }
    fprintf(f, "P5\n%d %d\n255\n", W, H);
    for (int i = 0; i < H * W; ++i) {
        float v = pixels[i];
        if (v < 0.0f)
            v = 0.0f;
        if (v > 1.0f)
            v = 1.0f;
        unsigned char c = (unsigned char)(v * 255.0f);
        fwrite(&c, 1, 1, f);
    }
    fclose(f);
}

static inline void save_pgm_grid(const char* path, const float* images,
    int n, int H, int W, int cols)
{
    int rows = (n + cols - 1) / cols;
    int GH = rows * (H + 1) - 1;
    int GW = cols * (W + 1) - 1;
    float* grid = (float*)calloc((size_t)GH * GW, sizeof(float));
    for (int i = 0; i < n; ++i) {
        int r = i / cols, c = i % cols;
        int y0 = r * (H + 1), x0 = c * (W + 1);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                grid[(y0 + y) * GW + (x0 + x)] = images[i * H * W + y * W + x];
    }
    save_pgm(path, grid, GH, GW);
    free(grid);
    printf("  saved %s  (%d images, %dx%d grid)\n", path, n, cols, rows);
}

#endif /* PGM_IO_H */
