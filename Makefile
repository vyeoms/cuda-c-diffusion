NVCC = nvcc
CC   = gcc
B    = build
S    = src

# Real SASS for your GPU + PTX so the driver can JIT for future cards.
#   Orin sm_87 / compute_87, Xavier sm_72 / compute_72, Turing sm_75 / compute_75, ...
GENCODE = -gencode arch=compute_75,code=sm_75 \
          -gencode arch=compute_75,code=compute_75

NVCCFLAGS = $(GENCODE) -O3 --use_fast_math
CFLAGS    = -O3 -std=c11 -Wall -I/usr/local/cuda/include -I$(S)
LIBS      = -lcublas -lcurand -lm

# --- shared module sources (everything except main files) ---
SHARED_SRC = linear.c mp_linear.c residual.c cond_residual.c mp_fourier.c sinusoidal.c \
             learned_embed.c plain_residual.c layernorm.c groupnorm.c activation.c \
             sequential.c optimizer.c loss.c conv2d.c downsample.c upsample.c \
             attention.c cond_conv_residual.c conv_residual.c \
             unet.c ddpm_unet.c ema.c checkpoint.c
SHARED_OBJ = $(addprefix $(B)/,$(SHARED_SRC:.c=.o))
CU_OBJ     = $(B)/kernels.o

# --- original nn demo ---
nn: $(B)/main.o $(filter-out $(B)/conv2d.o $(B)/downsample.o $(B)/upsample.o $(B)/attention.o $(B)/cond_conv_residual.o $(B)/conv_residual.o $(B)/unet.o $(B)/ddpm_unet.o, $(SHARED_OBJ)) $(CU_OBJ)
	$(NVCC) $(GENCODE) $^ -o nn $(LIBS)

# --- EDM diffusion demo ---
edm_demo: $(B)/edm/main_edm.o $(SHARED_OBJ) $(CU_OBJ)
	$(NVCC) $(GENCODE) $^ -o edm_demo $(LIBS)

# --- DDPM diffusion demo ---
ddpm_demo: $(B)/ddpm/main_ddpm.o $(B)/ddpm/ddpm.o $(SHARED_OBJ) $(CU_OBJ)
	$(NVCC) $(GENCODE) $^ -o ddpm_demo $(LIBS)

# --- finite-difference gradient check (FP32, full module coverage) ---
gradcheck: $(B)/tests/gradcheck.o $(SHARED_OBJ) $(CU_OBJ)
	$(NVCC) $(GENCODE) $^ -o gradcheck $(LIBS)

# --- compilation rules ---
$(B)/%.o: $(S)/%.c $(S)/nn.h | $(B)
	$(CC) $(CFLAGS) -c $< -o $@

$(B)/main.o: main.c $(S)/nn.h | $(B)
	$(CC) $(CFLAGS) -c $< -o $@

$(B)/edm/%.o: edm/%.c $(S)/nn.h | $(B)/edm
	$(CC) $(CFLAGS) -c $< -o $@

$(B)/ddpm/%.o: ddpm/%.c $(S)/nn.h ddpm/ddpm.h | $(B)/ddpm
	$(CC) $(CFLAGS) -c $< -o $@

$(B)/linear.o $(B)/mp_linear.o $(B)/conv2d.o $(B)/attention.o: $(S)/gemm.h
$(B)/edm/main_edm.o: $(S)/mnist_io.h $(S)/pgm_io.h $(S)/ini.h
$(B)/ddpm/main_ddpm.o: $(S)/mnist_io.h $(S)/pgm_io.h $(S)/ini.h

$(B)/kernels.o: $(S)/kernels.cu $(S)/nn.h | $(B)
	$(NVCC) $(NVCCFLAGS) -I$(S) -c $< -o $@

$(B) $(B)/edm $(B)/ddpm $(B)/tests $(B)/diffusion $(B)/samplers:
	mkdir -p $@

clean:
	rm -rf $(B) nn edm_demo ddpm_demo gradcheck
