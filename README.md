# vt-transformer
面向边缘和端，基于C++开发的 Transformer 开发框架，支持推理和训练。

A new Transformer framework for edge computing, based on pure c++, supports inference and training.

- 基于 C++ 的极简的架构设计， 隔离 Transformer 核心计算与应用逻辑，独立运行无第三方库依赖
- 支持包括国产硬件在内的多种异构平台
- 支持 fp16, Q4(g128), PQ3(自研)等量化压缩计算，内置KV Cache优化管理
- 性价比优先，通过软件优化边缘和端测的大模型运行性能（吞吐和训推速度）
- 支持微调训练与推理，适配主流的开源模型
- 提供示范的 ChatBot ，RAG  , Agent 应用案例

## 1. 当前开发状态

### 以下是完成以及计划进行适配的算力平台

- [x] 中科曙光 DCU（海光） Z100/Z100L 
- [x] Nvidia CUDA 芯片 T4/GTX 3080/3090
- [x] Intel CPU / GPU (Xe) 
- [ ] 寒武纪
- [ ] 希姆计算
- [ ] ... 


### 以下是完成以及计划进行移植的开源大模型

- [x] Qwen-7b   模型权重下载 model/qwen-7b/README.md 
- [x] Qwen-14b  模型权重下载 model/qwen-14b/README.md
- [ ] Qwen-70b
- [ ] Qwen-VL
- [ ] Whisper2
- [ ] ChatGLM
- [ ] Baichuan
- [ ] LLAMA2
- [ ] ...

###  当前性能指标

* 测试环境： Nvidia T4 单卡，16G显存，65 TFLOPS，运行Qwen-7B模型，
* 测试目标： 测试批量用户模型在线推理能力，8 batch，1024 上下文，FP16 (Activity)
* 测试结果： 35 tokens/s

## 2. 技术架构说明

VT-Transformer架构上包括 TensorType 异构计算、 DAG 计算图执行引擎、 KV Cache管理、Tokenizer本地库等组件构成

### 代码架构
* tensortype/tensortype.cpp  TensorType 异构计算抽象组件
  * CUDA 计算实现
    * tensortype/cuda_tensor.cpp CUDA 设备实现
    * tensortype/cuda_kernels/  CUDA kernels 代码
  * 海光 DCU 计算实现
    * tensortype/dcu_tensor.cpp 海光 DCU 设备实现
    * tensortype/dcu_kernels/ DCU kernels代码
  * tensortype/dnnl_tensor.cpp Intel oneDNN 支持设备实现 
* tensortype/dag.cpp 高效计算图 DAG 执行引擎
* tensortype/nn_*.cpp  KV Cache管理, 算子定义
* tokenizer.com 本地 Tokenizer 库

### 说明文档（TODO）

* [DAG 语法说明](docs/DAG_reference.md)
* 如何适配一个新的硬件平台
* 如何移植一个新的 Transformer 模型

## 3. 编译与开发测试 （TODO）

### 3.1 编译 tokenizer_all 组件

### 3.2 编译 Tensortype 组件

### 3.3 编译测试应用

## 4. 参考应用说明 （TODO）

## 5. 关于未开源的部分以及训推一体机产品 （TODO）