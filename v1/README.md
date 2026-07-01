# VT-Transformer
A Transformer computing framework for edge, based on pure c++, supports inference and training.

## Features

- High-Performance Tensor Computation 
    - Tensortype library: A lightweight C++ tensor library supporting mixed precision computing (F32, F16, BF16, Q8, Q4, PQ) on diverse hardware backends (CUDA, OpenCL, x86, ARM64).
- Efficient DAG Engine
    - A Flexible IR Engine: Utilizes a human-readable and optimizable macro-expansion based intermediate representation (IR) format for efficient DAG (Directed Acyclic Graph) execution via Just-In-Time (JIT) compilation.
- All in one library
    - A C++ tokenizer combo library.
    - KV-Cache & Batch Processing: Built-in KV-cache and continuous batch inference capabilities for faster and more efficient model inference.
    - HTTP/Chatbot/Finetue Integration: Offers native support for developing chatbot and HTTP-based applications.
    - QWen & LLAMA Family Compatibility: Seamlessly works with QWen-LLM, Qwen-VL, and LLAMA3-LLM language model families.


More info ：https://www.viitrix.com/