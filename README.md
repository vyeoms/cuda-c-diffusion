# cuda-c-diffusion

A from-scratch CUDA C implementation of diffusion models, built as a hackable research base. The only dependencies are CUDA/cuBLAS/cuRAND.

Two complete diffusion pipelines are implemented:

- **EDM** (Karras et al.): Magnitude-preserving UNet with MP convolutions, MP attention, FiLM conditioning, Fourier time embeddings, and Heun ODE sampling.
- **DDPM** (Ho et al.): Standard UNet with GroupNorm, sinusoidal time embeddings, additive conditioning, x0-prediction with min-SNR-gamma weighting, and DDIM sampling.

Both include EMA, checkpointing, and the Karras inverse-sqrt LR schedule. Configs are `.ini` files.

## Building and running

Requires CUDA toolkit (tested on sm_75 / Turing). Edit the `GENCODE` line in `Makefile` for your GPU.

```
python fetch_mnist.py          # download MNIST into data/
make edm_demo && ./edm_demo    # train + sample EDM
make ddpm_demo && ./ddpm_demo  # train + sample DDPM
```

Samples are written to `samples/` as PGM images.

## Directory structure

```
.
├── Makefile
├── main.c                   # minimal demo (conditioned MLP)
├── src/                     # shared library
│   ├── nn.h                 # types, context, module vtable, all declarations
│   ├── kernels.cu           # CUDA kernels (activations, norms, reductions, MP ops)
│   ├── gemm.h               # cuBLAS GEMM wrapper (header-only)
│   ├── mnist_io.h           # MNIST loader (header-only)
│   ├── pgm_io.h             # PGM image writer (header-only)
│   ├── ini.h                # .ini config parser (header-only)
│   ├── linear.c             # dense layer
│   ├── mp_linear.c          # magnitude-preserving dense (weight norm + gain)
│   ├── conv2d.c             # im2col convolution (NHWC)
│   ├── attention.c          # multi-head self-attention
│   ├── layernorm.c, groupnorm.c, activation.c
│   ├── cond_conv_residual.c # EDM residual block (MP ops, FiLM)
│   ├── conv_residual.c      # DDPM residual block (GroupNorm, additive cond)
│   ├── unet.c               # EDM UNet (MP skip connections)
│   ├── ddpm_unet.c          # DDPM UNet
│   ├── downsample.c, upsample.c
│   ├── mp_fourier.c         # random Fourier features (EDM2)
│   ├── sinusoidal.c         # sinusoidal positional embedding
│   ├── sequential.c         # Sequential container + ParamList
│   ├── optimizer.c          # SGD, Adam, inverse-sqrt LR schedule
│   ├── ema.c                # exponential moving average (device-only swap)
│   ├── checkpoint.c         # binary save/load
│   └── ...                  # residual.c, plain_residual.c, learned_embed.c, etc.
├── edm/
│   ├── main_edm.c           # EDM training + Heun sampling
│   └── edm.ini
└── ddpm/
    ├── main_ddpm.c          # DDPM training + DDIM sampling
    ├── ddpm.c               # noise schedule, DDIM chain
    ├── ddpm.h
    └── ddpm.ini
```

## Design

Modules follow a vtable pattern (`Module` struct with `forward`/`backward`/`parameters`/`free` function pointers). A `Context` carries the cuBLAS handle, stream, cuRAND generator, and an array of conditioning slots (`CondSlot`) for routing embeddings to residual blocks.

All spatial data uses NHWC layout. Convolutions use im2col + cuBLAS GEMM. The EDM path uses magnitude-preserving operations throughout (weight normalization, forced unit-norm columns, gain scalars, `mp_sum`/`mp_cat` for skip connections).
