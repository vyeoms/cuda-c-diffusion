# cuda-c-diffusion

A from-scratch CUDA C implementation of diffusion models, built as a hackable research base. The only dependencies are CUDA/cuBLAS/cuRAND.

## Building and running

Requires CUDA toolkit (tested on sm_75 / Turing). Edit the `GENCODE` line in `Makefile` for your GPU.

```
python fetch_mnist.py          # download MNIST into data/
make edm_demo && ./edm_demo    # train + sample EDM
make ddpm_demo && ./ddpm_demo  # train + sample DDPM
make gradcheck && ./gradcheck  # finite-difference check of every module's backward
```

Samples are written to `samples/` as PGM images. Each demo takes an optional `.ini` path (e.g. `./edm_demo edm/edm.ini`).

### Config keys

| key | meaning |
|-----|---------|
| `seed` | seeds both `rand()` (host) and cuRAND |
| `tf32` | 1 = cuBLAS TF32; 0 = full FP32 |
| `save_every` | checkpoint the full training state every N steps, 0 = off |
| `ckpt_dir` | directory for `step_<n>.ckpt` (default `checkpoints`) |
| `resume` | path to a `.ckpt` to resume from (restores model/EMA/Adam/step) |

A checkpoint stores model weights, the EMA shadow, Adam moments, and the step counter.

## Directory structure

The layering follows EDM's separation of concerns: a generic NN library, a
**denoiser** interface each diffusion framework implements (model + loss),
**samplers** that consume any denoiser, and thin per-pipeline entry points.

```
.
├── Makefile
├── main.c                   # minimal demo (conditioned MLP)
├── src/                     # NN library
│   ├── nn.h                 # types, context, module vtable
│   ├── kernels.cu           # CUDA kernels
│   ├── gemm.h, mnist_io.h, pgm_io.h, ini.h   # header-only utils
│   ├── linear.c, mp_linear.c, conv2d.c, attention.c
│   ├── layernorm.c, groupnorm.c, activation.c, downsample.c, upsample.c
│   ├── cond_conv_residual.c # EDM residual block
│   ├── conv_residual.c      # DDPM residual block
│   ├── unet.c               # EDM UNet (MP skips); ddpm_unet.c (GroupNorm UNet)
│   ├── mp_fourier.c, sinusoidal.c            # time embeddings
│   ├── sequential.c, optimizer.c, ema.c, loss.c, checkpoint.c
│   └── ...
├── diffusion/               # diffusion framework
│   ├── denoiser.h           # Denoiser interface: forward(x→x0, levels) / backward
│   ├── train.{c,h}          # generic training loop (LR schedule, EMA, checkpoint)
│   ├── edm.{c,h}            # EDM denoiser (preconditioner) + EDM loss
│   └── ddpm.{c,h}           # DDPM schedule + x0-prediction denoiser + min-SNR loss
├── samplers/                # samplers, each takes a Denoiser*
│   ├── heun.{c,h}           # EDM probability-flow ODE (Heun)
│   └── ddim.{c,h}           # deterministic DDIM
├── tests/
│   └── gradcheck.c          # finite-difference check of every module's backward
├── edm/   { main_edm.c, edm.ini }     # EDM entry point (orchestration only)
└── ddpm/  { main_ddpm.c, ddpm.ini }   # DDPM entry point (orchestration only)
```

## Design

**NN modules** follow a vtable pattern (`Module` struct with
`forward`/`backward`/`parameters`/`free` function pointers). A `Context` carries
the cuBLAS handle, stream, cuRAND generator, and conditioning slots (`CondSlot`)
for routing embeddings to residual blocks.

**Diffusion frameworks** are factored after Karras et al.'s EDM. A `Denoiser`
(`diffusion/denoiser.h`) maps a noisy input to an x0 estimate at a given noise
level — for EDM that's the preconditioned `D = c_skip·x + c_out·F`; for DDPM it's
the x0-prediction network. Every sampler and the training loss treat the model as
this black-box denoiser, so model / loss / sampler stay independently swappable.
Each framework co-locates its denoiser and loss (`diffusion/edm.c`,
`diffusion/ddpm.c`); `diffusion/train.c` is a generic loop shared by both.

All spatial data uses NHWC layout. Convolutions use im2col + cuBLAS GEMM. The EDM
path uses magnitude-preserving operations throughout (weight normalization,
forced unit-norm columns, gain scalars, `mp_sum`/`mp_cat` for skip connections).

Every backward pass is written by hand, so `tests/gradcheck.c` finite-difference
checks each module (run `make gradcheck`); add a `check_module(...)` line when you
add a layer.
