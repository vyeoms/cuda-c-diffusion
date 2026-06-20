#include "nn.h"
#include <math.h>
#include <stdlib.h>

typedef struct {
    int dim, max_batch;
    float wa, wb;
    Sequential* branch;
    float* scaled_g;
} ResidualState;

static void residual_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    ResidualState* st = (ResidualState*)self->state;
    sequential_forward(st->branch, in, batch, ctx);
    const float* y = sequential_output(st->branch);
    /* out = wa*x + wb*F(x) */
    launch_weighted_sum(out, in, y, st->wa, st->wb, batch * st->dim, ctx->stream);
}

static void residual_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)in;
    (void)out;
    ResidualState* st = (ResidualState*)self->state;
    int n = batch * st->dim;

    /* The branch sees wb*grad_out as its upstream gradient so its parameter
       gradients are scaled correctly (scaled_g = wb*grad_out + 0). */
    launch_weighted_sum(st->scaled_g, grad_out, grad_out, st->wb, 0.0f, n, ctx->stream);
    sequential_backward(st->branch, st->scaled_g, batch, ctx);
    const float* branch_grad_in = sequential_input_grad(st->branch);

    /* grad wrt input = skip path (wa*grad_out) + branch path (already wb-scaled) */
    launch_weighted_sum(grad_in, grad_out, branch_grad_in, st->wa, 1.0f, n, ctx->stream);
}

static void residual_parameters(Module* self, ParamList* pl)
{
    ResidualState* st = (ResidualState*)self->state;
    sequential_parameters(st->branch, pl);
}

static void residual_free(Module* self)
{
    ResidualState* st = (ResidualState*)self->state;
    sequential_free(st->branch);
    cudaFree(st->scaled_g);
    free(st);
    free(self);
}

Module* residual_create(Module** branch, int n, int max_batch, float t)
{
    int dim = branch[0]->in_dim;
    if (branch[n - 1]->out_dim != dim) {
        fprintf(stderr, "residual_create: branch must map dim->dim (%d != %d)\n",
            dim, branch[n - 1]->out_dim);
        exit(1);
    }
    Module* m = (Module*)malloc(sizeof(Module));
    ResidualState* st = (ResidualState*)malloc(sizeof(ResidualState));
    st->dim = dim;
    st->max_batch = max_batch;

    float d = sqrtf((1.0f - t) * (1.0f - t) + t * t);
    st->wa = (1.0f - t) / d;
    st->wb = t / d;

    st->branch = sequential_create(branch, n, max_batch);
    st->scaled_g = nn_device_alloc((size_t)max_batch * dim);

    m->name = "residual";
    m->in_dim = dim;
    m->out_dim = dim;
    m->state = st;
    m->forward = residual_forward;
    m->backward = residual_backward;
    m->parameters = residual_parameters;
    m->free = residual_free;
    return m;
}
