#include "nn.h"

typedef struct {
    int dim;
    Activation act;
} ActState;

static void activation_forward(Module* self, const float* in, float* out,
    int batch, Context* ctx)
{
    ActState* st = (ActState*)self->state;
    launch_activation_forward(out, in, batch * st->dim, st->act, ctx->stream);
}

static void activation_backward(Module* self, const float* in, const float* out,
    const float* grad_out, float* grad_in,
    int batch, Context* ctx)
{
    (void)out;
    ActState* st = (ActState*)self->state;
    launch_activation_backward(grad_in, grad_out, in, batch * st->dim,
        st->act, ctx->stream);
}

static void activation_free(Module* self)
{
    free(self->state);
    free(self);
}

Module* activation_create(int dim, Activation act)
{
    Module* m = (Module*)malloc(sizeof(Module));
    ActState* st = (ActState*)malloc(sizeof(ActState));
    st->dim = dim;
    st->act = act;

    m->name = "activation";
    m->in_dim = dim;
    m->out_dim = dim;
    m->state = st;
    m->forward = activation_forward;
    m->backward = activation_backward;
    m->parameters = NULL;
    m->free = activation_free;
    return m;
}
