# core 测试构建
# 用法:
#   make        # 构建 test/mock
#   make run    # 构建并运行
#   make clean

CXX      ?= g++
CXXFLAGS := -std=c++17 -Wall -Wextra -g -O0
INCLUDES := -I../core

# CUDA 工具链: vt_cuda.cu 含定制 kernel, 由 nvcc device 编译
CUDA_PATH ?= /usr/local/cuda
NVCC      := $(CUDA_PATH)/bin/nvcc
NVCCFLAGS := -std=c++17 -Xcompiler -Wall,-Wextra -O2 -I../core -I$(CUDA_PATH)/include
LDLIBS   := -lcublasLt -lcublas

BIN   := mock
SRCS  := mock.cpp
CUSRC := ../core/scheduler_cuda.cpp \
         ../core/vt_cuda.cu \
         ../core/vt_cuda.cpp \
         ../core/vt.cpp

all: $(BIN)

$(BIN): $(SRCS) $(CUSRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS) $(CUSRC) $(LDLIBS)

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)

.PHONY: all run clean
