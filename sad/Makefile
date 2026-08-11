NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH ?= native
BUILD_DIR := build
TARGET ?= $(BUILD_DIR)/libsad_cuda.so
SOURCE := src/sad_cuda.cu
HEADER := include/sad_api.h
CUDA_HEADERS := $(shell find src -name '*.cuh' -print)

NVCCFLAGS := -O3 -std=c++17 -arch=$(CUDA_ARCH) --shared -Xcompiler=-fPIC \
	-lineinfo -Iinclude $(EXTRA_NVCCFLAGS)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE) $(HEADER) $(CUDA_HEADERS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SOURCE) -o $(TARGET)

clean:
	$(RM) $(TARGET)
