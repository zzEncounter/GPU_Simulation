NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH ?= native
CUQUANTUM_ROOT ?= /home/rzzhang/sad/.venv/lib/python3.12/site-packages/cuquantum
TARGET ?= build/libcuquantum_sad.so

NVCCFLAGS := -O2 -std=c++17 -arch=$(CUDA_ARCH) --shared -Xcompiler=-fPIC \
	-Iinclude -I$(CUQUANTUM_ROOT)/include
LDFLAGS := -L$(CUQUANTUM_ROOT)/lib -Xlinker -l:libcustatevec.so.1 \
	-Xlinker -rpath -Xlinker $(CUQUANTUM_ROOT)/lib

.PHONY: all clean

all: $(TARGET)

$(TARGET): src/cuquantum_cuda.cu include/cuquantum_api.h
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf build
