# Makefile — 构建 Qwen3-0.6B 推理服务（./bin/qwen3-server）。
#
# 用法：
#   make                # 构建到 ./bin/qwen3-server
#   make clean          # 删除 ./bin/
#   CUDA_PATH=/opt/cuda make    # 指定非默认 CUDA 安装路径
#
# 范围：本 Makefile 只负责把 main + qwen3 + core 库 + libzmq/libcudart 串成
# 一个可执行文件。推理逻辑（Qwen3Engine、forward kernel、命令行解析、
# Endpoint 主循环等）由 main.cpp / qwen3.cpp 自行实现——目前 main() 是空壳，
# make 也能产出 no-op 二进制；后续把代码补上即可。

CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O2 -Wall -Wextra -MMD -MP

# nvcc：编译 kernels/*.cu。SM_ARCH 可由环境覆盖（默认 sm_80 = A100/3090 等）。
NVCC      ?= nvcc
SM_ARCH   ?= 80
NVCCFLAGS ?= -std=c++17 -O2 -arch=sm_$(SM_ARCH) -Xcompiler -Wall -Xcompiler -Wextra -MMD -MP

# 项目根 = vt-transformer/vt-transformer/（从 models/qwen3-0.6b/ 上溯两级）。
ROOT      := ../..
CORE      := $(ROOT)/core
OPT       := $(ROOT)/opt
CUDA_PATH ?= /usr/local/cuda

INCLUDES  := -I$(ROOT) -I$(OPT) -I$(CUDA_PATH)/include
LDFLAGS   := -L$(CUDA_PATH)/lib64
LDLIBS    := -lcudart -lcublas -lcublasLt -lzmq -lpthread

BIN    := bin
TARGET := $(BIN)/qwen3-server

# 链接到 server 的源文件：
#   - main.cpp / qwen3.cpp           ：本目录（g++）
#   - kernels/prefill.cu             ：CUDA kernel（nvcc）
#   - core/{vt,vt_backend,vt_scheduler,vt_cuda}.cpp ：后端框架（g++）
OBJS := \
    $(BIN)/main.o \
    $(BIN)/qwen3.o \
    $(BIN)/kernels/prefill.o \
    $(BIN)/core/vt.o \
    $(BIN)/core/vt_backend.o \
    $(BIN)/core/vt_scheduler.o \
    $(BIN)/core/vt_cuda.o

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# 让 make 在 BIN 下保留源相对路径生成 .o（避免日后 core/ 与本目录同名文件冲突）。
# 本目录的 .cpp 编译到 $(BIN)/，core/ 下的 .cpp 编译到 $(BIN)/core/。
$(BIN)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# kernels/ 下的 .cu 用 nvcc 编译到 $(BIN)/kernels/。
$(BIN)/kernels/%.o: kernels/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BIN)/core/%.o: $(CORE)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(BIN)

-include $(OBJS:.o=.d)
