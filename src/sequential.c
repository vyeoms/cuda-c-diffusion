#include "nn.h"

/* ---------------- ParamList ---------------- */

void param_list_init(ParamList* pl)
{
    pl->count = 0;
    pl->capacity = 64;
    pl->items = (Param*)malloc(pl->capacity * sizeof(Param));
}

void param_list_add(ParamList* pl, float* param, float* grad, int size)
{
    if (pl->count == pl->capacity) {
        pl->capacity *= 2;
        pl->items = (Param*)realloc(pl->items, pl->capacity * sizeof(Param));
    }
    pl->items[pl->count].param = param;
    pl->items[pl->count].grad = grad;
    pl->items[pl->count].size = size;
    pl->count++;
}

void param_list_free(ParamList* pl)
{
    free(pl->items);
    pl->items = NULL;
    pl->count = pl->capacity = 0;
}

/* ---------------- Sequential ----------------

   acts[0..n]   : acts[0] = input copy, acts[i+1] = output of module i
   grads[0..n-1]: grads[i] = grad wrt acts[i] (grads[0] = grad wrt input)
   The top-of-stack gradient (wrt acts[n]) comes in from the loss head.      */

struct Sequential {
    Module** mods;
    int n;
    int max_batch;
    int* dims; /* n+1: dims[0]=mods[0].in_dim, dims[i]=mods[i-1].out_dim */
    float** acts; /* n+1 */
    float** grads; /* n   */
};

Sequential* sequential_create(Module** modules, int n, int max_batch)
{
    Sequential* s = (Sequential*)malloc(sizeof(Sequential));
    s->n = n;
    s->max_batch = max_batch;
    s->mods = (Module**)malloc(n * sizeof(Module*));
    for (int i = 0; i < n; ++i)
        s->mods[i] = modules[i];

    s->dims = (int*)malloc((n + 1) * sizeof(int));
    s->dims[0] = modules[0]->in_dim;
    for (int i = 1; i <= n; ++i)
        s->dims[i] = modules[i - 1]->out_dim;
    for (int i = 1; i < n; ++i) {
        if (modules[i]->in_dim != modules[i - 1]->out_dim) {
            fprintf(stderr, "dim mismatch: %s.out=%d but %s.in=%d\n",
                modules[i - 1]->name, modules[i - 1]->out_dim,
                modules[i]->name, modules[i]->in_dim);
            exit(1);
        }
    }

    s->acts = (float**)malloc((n + 1) * sizeof(float*));
    s->grads = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i <= n; ++i)
        s->acts[i] = nn_device_alloc((size_t)max_batch * s->dims[i]);
    for (int i = 0; i < n; ++i)
        s->grads[i] = nn_device_alloc((size_t)max_batch * s->dims[i]);
    return s;
}

void sequential_forward(Sequential* s, const float* input, int batch, Context* ctx)
{
    NN_CUDA_CHECK(cudaMemcpyAsync(s->acts[0], input,
        (size_t)batch * s->dims[0] * sizeof(float),
        cudaMemcpyDeviceToDevice, ctx->stream));
    for (int i = 0; i < s->n; ++i)
        s->mods[i]->forward(s->mods[i], s->acts[i], s->acts[i + 1], batch, ctx);
}

void sequential_backward(Sequential* s, const float* grad_output, int batch, Context* ctx)
{
    for (int i = s->n - 1; i >= 0; --i) {
        const float* go = (i == s->n - 1) ? grad_output : s->grads[i + 1];
        s->mods[i]->backward(s->mods[i], s->acts[i], s->acts[i + 1],
            go, s->grads[i], batch, ctx);
    }
}

float* sequential_output(Sequential* s) { return s->acts[s->n]; }
float* sequential_input_grad(Sequential* s) { return s->grads[0]; }

void sequential_parameters(Sequential* s, ParamList* pl)
{
    for (int i = 0; i < s->n; ++i)
        if (s->mods[i]->parameters)
            s->mods[i]->parameters(s->mods[i], pl);
}

void sequential_free(Sequential* s)
{
    for (int i = 0; i < s->n; ++i)
        s->mods[i]->free(s->mods[i]);
    for (int i = 0; i <= s->n; ++i)
        cudaFree(s->acts[i]);
    for (int i = 0; i < s->n; ++i)
        cudaFree(s->grads[i]);
    free(s->mods);
    free(s->dims);
    free(s->acts);
    free(s->grads);
    free(s);
}
