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
# -rdc=true：开可重定位 device code，让 prefill.cu 里的 prefill_kernel 能跨 TU
#   调用 embedding.cu / rmsnorm.cu 里定义的 __device__ 函数；不开 ptxas 会报
#   "Unresolved extern function"。开了 -rdc 后必须配一次 nvcc -dlink（见下方
#   kernels_dlink.o 规则）并在最终 link 加 -lcudadevrt。
NVCC      ?= nvcc
SM_ARCH   ?= 75
NVCCFLAGS ?= -std=c++17 -O2 -arch=sm_$(SM_ARCH) -rdc=true -Xcompiler -Wall -Xcompiler -Wextra -MMD -MP

# 项目根 = vt-transformer/vt-transformer/（从 models/qwen3-0.6b/ 上溯两级）。
ROOT      := ../..
CORE      := $(ROOT)/core
OPT       := $(ROOT)/opt
CUDA_PATH ?= /usr/local/cuda

INCLUDES  := -I$(ROOT) -I$(OPT) -I$(CUDA_PATH)/include
LDFLAGS   := -L$(CUDA_PATH)/lib64
LDLIBS    := -lcudart -lcudadevrt -lcublas -lcublasLt -lzmq -lpthread

BIN    := bin
TARGET := $(BIN)/qwen3-server

# 链接到 server 的源文件：
#   - main.cpp / qwen3.cpp                                  ：本目录（g++）
#   - kernels/{prefill,embedding,rmsnorm,qkv_proj}.cu       ：CUDA kernel（nvcc）
#   - core/{vt,vt_backend,vt_scheduler,vt_cuda}.cpp         ：后端框架（g++）
OBJS := \
    $(BIN)/main.o \
    $(BIN)/qwen3.o \
    $(BIN)/kernels/prefill.o \
    $(BIN)/kernels/embedding.o \
    $(BIN)/kernels/rmsnorm.o \
    $(BIN)/kernels/qkv_proj.o \
    $(BIN)/core/vt.o \
    $(BIN)/core/vt_backend.o \
    $(BIN)/core/vt_scheduler.o \
    $(BIN)/core/vt_cuda.o

# 用 -rdc=true 编译的 .o 列表 —— device link 阶段需要把它们的 device 代码合并成
# 一个 kernels_dlink.o，否则 host linker 仍然解析不了跨 TU 的 __device__ 符号。
CUDA_OBJS := \
    $(BIN)/kernels/prefill.o \
    $(BIN)/kernels/embedding.o \
    $(BIN)/kernels/rmsnorm.o \
    $(BIN)/kernels/qkv_proj.o

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJS) $(BIN)/kernels_dlink.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Device link：把 CUDA_OBJS 里所有 device code 合到一个 .o 里，host link 时
# 跟其他 .o 一起塞进最终 binary。规则放在 $(TARGET) 之后，避免被当成默认目标。
$(BIN)/kernels_dlink.o: $(CUDA_OBJS)
	$(NVCC) -arch=sm_$(SM_ARCH) -dlink $^ -o $@

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
