# Makefile — build main.cpp + weights.cpp + schedule.cpp for the Qwen3-0.6B backend
#
# Usage:
#   make              # build ./qwen3_infer (CUDA backend)
#   make run          # build + run
#   make run MODEL=/path/to/qwen3-0.6b
#   make clean

CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O2 -Wall -Wextra

ROOT_DIR  := $(CURDIR)/../..
OPT_DIR   := $(ROOT_DIR)/opt
CORE_DIR  := $(ROOT_DIR)/core

# CUDA 12.6 在 /usr/local/cuda（标准软链）。
# 覆盖方式：make CUDA_HOME=/path/to/cuda
CUDA_HOME ?= /usr/local/cuda

INCLUDES  := -I$(CORE_DIR) -I$(ROOT_DIR) -I$(OPT_DIR) -I$(CUDA_HOME)/include

TARGET  := qwen3_infer
SRCS    := main.cpp weights.cpp qwen3_cuda_backend.cpp \
           $(CORE_DIR)/schedule.cpp $(CORE_DIR)/vt_cuda.cpp $(CORE_DIR)/vt.cpp
LDFLAGS := -lzmq -L$(CUDA_HOME)/lib64 -lcudart -lcublasLt -lcublas

# 默认模型目录（main.cpp 的 argv[1] 也接受这个）
MODEL   ?= /home/teaonly/workspace/qwen3-0.6b

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -DMINISGL_USE_CUDA $(INCLUDES) $(SRCS) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) $(MODEL)

clean:
	rm -f $(TARGET)
