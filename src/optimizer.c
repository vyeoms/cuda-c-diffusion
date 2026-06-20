#include "nn.h"
#include <math.h>

typedef enum { OPT_SGD = 0,
    OPT_ADAM } OptKind;

struct Optimizer {
    OptKind kind;
    float lr, beta1, beta2, eps;
    int t;
    int n; /* number of parameter tensors */
    float **m, **v; /* Adam moments per tensor (device); NULL for SGD */
};

Optimizer* optimizer_sgd(const ParamList* pl, float lr)
{
    Optimizer* o = (Optimizer*)malloc(sizeof(Optimizer));
    o->kind = OPT_SGD;
    o->lr = lr;
    o->n = pl->count;
    o->t = 0;
    o->m = NULL;
    o->v = NULL;
    return o;
}

Optimizer* optimizer_adam(const ParamList* pl, float lr,
    float beta1, float beta2, float eps)
{
    Optimizer* o = (Optimizer*)malloc(sizeof(Optimizer));
    o->kind = OPT_ADAM;
    o->lr = lr;
    o->beta1 = beta1;
    o->beta2 = beta2;
    o->eps = eps;
    o->t = 0;
    o->n = pl->count;
    o->m = (float**)malloc(o->n * sizeof(float*));
    o->v = (float**)malloc(o->n * sizeof(float*));
    for (int i = 0; i < o->n; ++i) {
        size_t sz = (size_t)pl->items[i].size;
        o->m[i] = nn_device_alloc(sz);
        o->v[i] = nn_device_alloc(sz);
        NN_CUDA_CHECK(cudaMemset(o->m[i], 0, sz * sizeof(float)));
        NN_CUDA_CHECK(cudaMemset(o->v[i], 0, sz * sizeof(float)));
    }
    return o;
}

void optimizer_step(Optimizer* o, const ParamList* pl, Context* ctx)
{
    if (o->kind == OPT_SGD) {
        for (int i = 0; i < pl->count; ++i)
            launch_sgd_step(pl->items[i].param, pl->items[i].grad,
                o->lr, pl->items[i].size, ctx->stream);
        return;
    }
    o->t++;
    float bc1 = 1.0f - powf(o->beta1, (float)o->t);
    float bc2 = 1.0f - powf(o->beta2, (float)o->t);
    for (int i = 0; i < pl->count; ++i)
        launch_adam_step(pl->items[i].param, o->m[i], o->v[i], pl->items[i].grad,
            o->lr, o->beta1, o->beta2, o->eps, bc1, bc2,
            pl->items[i].size, ctx->stream);
}

void optimizer_set_lr(Optimizer* o, float lr) { o->lr = lr; }

void optimizer_free(Optimizer* o)
{
    if (o->m) {
        for (int i = 0; i < o->n; ++i) {
            cudaFree(o->m[i]);
            cudaFree(o->v[i]);
        }
        free(o->m);
        free(o->v);
    }
    free(o);
}
