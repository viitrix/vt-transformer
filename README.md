# VT-Transformer
面向边缘和端，基于C++开发的 Transformer 开发框架，支持推理和训练。

A new Transformer framework for edge computing, based on pure c++, supports inference and training.

- 基于 C++ 的极简架构设计， 隔离 Transformer 核心计算与应用逻辑，独立运行无第三方库依赖
- 支持包括国产硬件在内的异构平台
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

- [x] Qwen-1.8b 
- [x] Qwen-7b  (模型文件下载：链接：https://pan.baidu.com/s/1R3kNzM4CaDcFrFw_UXYxhA 提取码：qmta )
- [x] Qwen-14b 
- [x] Qwen-72b
- [ ] Qwen-VL
- [x] Whisper2
- [ ] ChatGLM
- [x] Baichuan
- [ ] LLAMA2
- [ ] ...

注意，需要下载更多转换好的模型文件，可以联系我们的技术支持，参考 https://www.viitrix.com/ 。

###  当前性能指标

* 测试环境： Nvidia T4 单卡，16G显存，65 TFLOPS，运行Qwen-7B模型，
* 测试目标： 测试批量用户模型在线推理能力，8 batch，1024 上下文，FP16 (Activity)
* 测试结果： 35 tokens/s

## 2. 技术架构说明

VT-Transformer 架构上包括 TensorType 异构计算、 DAG 计算图执行引擎、 KV Cache 管理、Tokenizer组合库等组件构成。

### 代码架构
* tensortype/tensortype.cpp(.hpp)  TensorType 异构计算抽象组件
  * CUDA 计算实现
    * tensortype/cuda_tensor.cpp(.hpp) CUDA 设备实现
    * tensortype/cuda_kernels/  CUDA kernels 代码
  * 海光 DCU 计算实现
    * tensortype/dcu_tensor.cpp(.hpp) 海光 DCU 设备实现
    * tensortype/dcu_kernels/ DCU kernels代码
  * tensortype/dnnl_tensor.cpp(.hpp) Intel oneDNN 支持设备实现 
* tensortype/dag.cpp(.hpp) 高效计算图 DAG 执行引擎
* tensortype/nn_*.*  KV Cache管理, 算子定义
* tools/ 提供了一个 repl ，tokenizer测试
* models 下提供了不同模型的 DAG 文件，简单的终端 ChatBot，以及性能测试代码 
* tokenizer.com 本地 Tokenizer 库

### 开发文档

* [DAG 语法说明](docs/DAG_reference.md)
* [如何适配一个新的硬件平台](docs/new_platform.md)
* [如何移植一个新的 Transformer 模型](docs/porting_model.md)

## 3. 编译与开发测试 

在根目录下有有一个build_env.sh ，里面修改相应的开发环境依赖路径，如 CUDA/DTK/DNNL 等路径。

### 3.1 编译 tokenizers.combo 组件

VT-Transformer集成了 tokenizers (rust/openai), tiktoken(rust/huggingface), sentencepiece(c++/google) 三个最常用的Tokenizer库的封装。
参考 tokenizers.combo/Makefile 的编译实现。

### 3.2 编译 Tensortype 组件

主要参考 tensortype/Makefile 的编译实现，一个平台对应一个编译文件，这些构建文件非常简洁，方便修改调试。

### 3.3 编译测试应用

models目录下的有对应的Makefile文件，非常简单，可以参考修改调整。目前测试应用主要包括终端会话的chatbot, 以及一个简单的性能测试。

## 4. 参考应用说明 

完整的演示应用（RAG / Agent）计划后续提供。

## 5. 关于训推一体机产品 

上海云锦微基于VT-Transformer（开源）开发了完整的训推一体机，具体参考：https://www.viitrix.com/ 。
