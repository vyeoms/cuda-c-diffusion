#include "nn.h"
#include <stdlib.h>

typedef struct {
    int dim, max_batch;
    Sequential* branch;
} PlainResState;

static void plain_residual_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    PlainResState* st = (PlainResState*)self->state;
    sequential_forward(st->branch, in, batch, ctx);
    const float* y = sequential_output(st->branch);
    /* out = x + F(x) */
    launch_weighted_sum(out, in, y, 1.0f, 1.0f, batch * st->dim, ctx->stream);
}

static void plain_residual_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)in;
    (void)out;
    PlainResState* st = (PlainResState*)self->state;
    sequential_backward(st->branch, grad_out, batch, ctx);
    const float* branch_grad_in = sequential_input_grad(st->branch);
    launch_weighted_sum(grad_in, grad_out, branch_grad_in, 1.0f, 1.0f,
        batch * st->dim, ctx->stream);
}

static void plain_residual_parameters(Module* self, ParamList* pl)
{
    PlainResState* st = (PlainResState*)self->state;
    sequential_parameters(st->branch, pl);
}

static void plain_residual_free(Module* self)
{
    PlainResState* st = (PlainResState*)self->state;
    sequential_free(st->branch);
    free(st);
    free(self);
}

Module* plain_residual_create(Module** branch, int n, int max_batch)
{
    int dim = branch[0]->in_dim;
    if (branch[n - 1]->out_dim != dim) {
        fprintf(stderr, "plain_residual_create: branch must map dim->dim (%d != %d)\n",
            dim, branch[n - 1]->out_dim);
        exit(1);
    }
    Module* m = (Module*)malloc(sizeof(Module));
    PlainResState* st = (PlainResState*)malloc(sizeof(PlainResState));
    st->dim = dim;
    st->max_batch = max_batch;
    st->branch = sequential_create(branch, n, max_batch);

    m->name = "plain_residual";
    m->in_dim = dim;
    m->out_dim = dim;
    m->state = st;
    m->forward = plain_residual_forward;
    m->backward = plain_residual_backward;
    m->parameters = plain_residual_parameters;
    m->free = plain_residual_free;
    return m;
}
