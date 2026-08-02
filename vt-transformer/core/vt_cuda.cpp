#include "vt_cuda.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include <cuda_fp16.h>

namespace vt {

namespace op {

struct Create : public NativeWord {
    void run(Stack& stack) override {
        auto device = stack.pop_string();
        size_t bytes = (size_t)stack.pop_number();

        tensor_t t;
        if ( device == "cuda" ) {
            t = std::make_shared<CudaTensor>(CudaMemoryType::Device, bytes);
        } else if ( device == "host" ) {
            t = std::make_shared<CudaTensor>(CudaMemoryType::Host, bytes);
        } else {
            vt_fatal_error();
        }

        stack.push_tensor(t);
    }
    NWORD_CREATOR_DEFINE_CTX(Create)
};

// cuda.view  : tensor offset size -- tensor'
// 非拥有型 view，按 [offset, offset+size) 切源 tensor；source 由 source_ 字段保活
struct View : public NativeWord {
    void run(Stack& stack) override {
        auto view_size = (uint64_t)stack.pop_number();
        auto offset    = (uint64_t)stack.pop_number();
        auto src       = stack.pop_tensor();

        auto t = std::make_shared<CudaTensor>(src, offset, view_size);
        stack.push_tensor(t);
    }
    NWORD_CREATOR_DEFINE_CTX(View)
};

// cuda.to_host : src dst -- dst
// 把 src 的字节拷到 dst；dst 必须是 host tensor，大小必须与 src 一致。
// 不创建新 tensor；dst 引用计数 +1 后推回栈（共享同一个底层对象）。
// 走 ctx 的 current_stream，需要可见性时显式 cuda.sync。
struct ToHost : public NativeWord {
    void run(Stack& stack) override {
        auto dst = stack.pop_tensor();
        auto src = stack.pop_tensor();
        auto* cuda_src = dynamic_cast<CudaTensor*>(src.get());
        auto* cuda_dst = dynamic_cast<CudaTensor*>(dst.get());
        if (!cuda_src || !cuda_dst) vt_fatal_error();
        if (!cuda_dst->is_host())
            vt_panic("cuda.to_host: dst must be host tensor");
        if (cuda_src->size() != cuda_dst->size())
            vt_panic("cuda.to_host: src/dst size mismatch");
        if (cuda_src->size() > 0) {
            auto* cctx = dynamic_cast<CudaContext*>(ctx_);
            if (!cctx) vt_fatal_error();
            CUDA_CHECK(cudaMemcpyAsync(cuda_dst->data(), cuda_src->data(),
                                       cuda_src->size(), cudaMemcpyDefault,
                                       cctx->current_stream()));
        }
        stack.push_tensor(dst);
    }
    NWORD_CREATOR_DEFINE_CTX(ToHost)
};

// cuda.to_device : src dst -- dst
// 把 src 的字节拷到 dst；dst 必须是 device tensor，大小必须与 src 一致。
struct ToDevice : public NativeWord {
    void run(Stack& stack) override {
        auto dst = stack.pop_tensor();
        auto src = stack.pop_tensor();
        auto* cuda_src = dynamic_cast<CudaTensor*>(src.get());
        auto* cuda_dst = dynamic_cast<CudaTensor*>(dst.get());
        if (!cuda_src || !cuda_dst) vt_fatal_error();
        if (!cuda_dst->is_device())
            vt_panic("cuda.to_device: dst must be device tensor");
        if (cuda_src->size() != cuda_dst->size())
            vt_panic("cuda.to_device: src/dst size mismatch");
        if (cuda_src->size() > 0) {
            auto* cctx = dynamic_cast<CudaContext*>(ctx_);
            if (!cctx) vt_fatal_error();
            CUDA_CHECK(cudaMemcpyAsync(cuda_dst->data(), cuda_src->data(),
                                       cuda_src->size(), cudaMemcpyDefault,
                                       cctx->current_stream()));
        }
        stack.push_tensor(dst);
    }
    NWORD_CREATOR_DEFINE_CTX(ToDevice)
};

// cuda.zero : tensor -- tensor
// 就地清零。device 分支走 cudaMemsetAsync（current_stream），host 分支走 std::memset。
// tensor 推回栈方便链式调用。
struct Zero : public NativeWord {
    void run(Stack& stack) override {
        auto t = stack.pop_tensor();
        auto* ct = dynamic_cast<CudaTensor*>(t.get());
        if (!ct) vt_fatal_error();
        if (ct->size() > 0) {
            if (ct->is_device()) {
                auto* cctx = dynamic_cast<CudaContext*>(ctx_);
                if (!cctx) vt_fatal_error();
                CUDA_CHECK(cudaMemsetAsync(ct->data(), 0, ct->size(),
                                           cctx->current_stream()));
            } else {
                std::memset(ct->data(), 0, ct->size());
            }
        }
        stack.push_tensor(t);
    }
    NWORD_CREATOR_DEFINE_CTX(Zero)
};

// cuda.load : tensor path --
// 把 path 的全部二进制内容读进 tensor；tensor 的 mem_type / 大小由调用方预先决定，
// 文件字节数必须与 tensor 大小一致，否则 panic。tensor 不推回栈。
// device 分支走 current_stream 异步拷贝（需要可见性时显式 cuda.sync）。
struct Load : public NativeWord {
    void run(Stack& stack) override {
        auto path = stack.pop_string();
        auto t    = stack.pop_tensor();
        auto* ct  = dynamic_cast<CudaTensor*>(t.get());
        if (!ct) vt_fatal_error();

        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            std::string msg = "cuda.load: cannot open file: " + path;
            vt_panic(msg.c_str());
        }

        auto end = f.tellg();
        if (end < 0) vt_panic("cuda.load: tellg failed");
        size_t bytes = static_cast<size_t>(end);
        if (bytes != ct->size())
            vt_panic("cuda.load: file size != tensor size");
        f.seekg(0, std::ios::beg);

        if (bytes > 0) {
            if (ct->is_host()) {
                f.read(static_cast<char*>(ct->data()), static_cast<std::streamsize>(bytes));
                if (static_cast<size_t>(f.gcount()) != bytes)
                    vt_panic("cuda.load: short read");
            } else {
                std::vector<char> buf(bytes);
                f.read(buf.data(), static_cast<std::streamsize>(bytes));
                if (static_cast<size_t>(f.gcount()) != bytes)
                    vt_panic("cuda.load: short read");
                auto* cctx = dynamic_cast<CudaContext*>(ctx_);
                if (!cctx) vt_fatal_error();
                CUDA_CHECK(cudaMemcpyAsync(ct->data(), buf.data(), bytes,
                                           cudaMemcpyHostToDevice,
                                           cctx->current_stream()));
            }
        }
    }
    NWORD_CREATOR_DEFINE_CTX(Load)
};

// cuda.size : tensor -- bytes
// 推 tensor 的字节大小（数字）到栈。tensor 被消费。
struct Size : public NativeWord {
    void run(Stack& stack) override {
        auto t = stack.pop_tensor();
        auto* ct = dynamic_cast<CudaTensor*>(t.get());
        if (!ct) vt_fatal_error();
        stack.push_number((double)ct->size());
    }
    NWORD_CREATOR_DEFINE_CTX(Size)
};

// cuda.is_host : tensor -- bool
// 推 1.0 (true) 或 0.0 (false)。
struct IsHost : public NativeWord {
    void run(Stack& stack) override {
        auto t = stack.pop_tensor();
        auto* ct = dynamic_cast<CudaTensor*>(t.get());
        if (!ct) vt_fatal_error();
        stack.push_number(ct->is_host() ? 1.0 : 0.0);
    }
    NWORD_CREATOR_DEFINE_CTX(IsHost)
};

// cuda.is_device : tensor -- bool
struct IsDevice : public NativeWord {
    void run(Stack& stack) override {
        auto t = stack.pop_tensor();
        auto* ct = dynamic_cast<CudaTensor*>(t.get());
        if (!ct) vt_fatal_error();
        stack.push_number(ct->is_device() ? 1.0 : 0.0);
    }
    NWORD_CREATOR_DEFINE_CTX(IsDevice)
};

// cuda.data_ptr : tensor -- number
// 取 tensor 底层 raw 指针，以 double 形式推回栈（指针经 uintptr_t 中转，
// 在 64-bit 下用户态指针 ≤ 48 bit，double 的 52-bit mantissa 装得下）。
// 用途：把 device 指针传给需要 raw ptr 的 native word（如 kernel launcher、
// qwen3.prefill 这类按 number 收指针的入口）。tensor 被消费。
// 仅对 CudaTensor 有意义，其它 tensor 类型 panic。
struct DataPtr : public NativeWord {
    void run(Stack& stack) override {
        auto t = stack.pop_tensor();
        auto* ct = dynamic_cast<CudaTensor*>(t.get());
        if (!ct) vt_fatal_error();
        stack.push_number(static_cast<double>(reinterpret_cast<uintptr_t>(ct->data())));
    }
    NWORD_CREATOR_DEFINE_CTX(DataPtr)
};

// cuda.dump : tensor type_str --
// 简短打印 tensor 内容：head 8 + tail 8 个元素；元素总数少则全打印。
// type_str 支持 "fp16" / "int32"。
// device tensor 会经 current_stream 同步拷贝到 host staging buffer 再打印，
// 因此调用返回时数据已就绪，无需额外 cuda.sync。
struct Dump : public NativeWord {
    void run(Stack& stack) override {
        auto type_str = stack.pop_string();
        auto t = stack.pop_tensor();
        auto* ct = dynamic_cast<CudaTensor*>(t.get());
        if (!ct) vt_fatal_error();

        size_t elem_size = 0;
        if      (type_str == "fp16")  elem_size = 2;
        else if (type_str == "int32") elem_size = 4;
        else vt_panic("cuda.dump: unsupported type, expect fp16/int32");

        size_t total_bytes = ct->size();
        if (total_bytes % elem_size != 0)
            vt_panic("cuda.dump: tensor size not multiple of elem_size");
        size_t n_elems = total_bytes / elem_size;

        std::vector<char> staging;
        const void* host_ptr = nullptr;
        if (ct->is_host()) {
            host_ptr = ct->data();
        } else {
            staging.resize(total_bytes);
            auto* cctx = dynamic_cast<CudaContext*>(ctx_);
            if (!cctx) vt_fatal_error();
            CUDA_CHECK(cudaMemcpyAsync(staging.data(), ct->data(), total_bytes,
                                       cudaMemcpyDeviceToHost,
                                       cctx->current_stream()));
            cctx->sync_current();
            host_ptr = staging.data();
        }

        const size_t k_head = 10;
        const size_t k_tail = 10;
        std::ostringstream oss;
        oss << "cuda.dump[" << type_str << "] n_elems=" << n_elems << ":";

        auto print_elem = [&](size_t i) {
            if (type_str == "fp16") {
                const __half* p = reinterpret_cast<const __half*>(host_ptr) + i;
                oss << " " << static_cast<double>(__half2float(*p));
            } else {
                oss << " " << (reinterpret_cast<const int32_t*>(host_ptr))[i];
            }
        };

        if (n_elems <= k_head + k_tail) {
            for (size_t i = 0; i < n_elems; ++i) print_elem(i);
        } else {
            for (size_t i = 0; i < k_head; ++i) print_elem(i);
            oss << " ...";
            for (size_t i = n_elems - k_tail; i < n_elems; ++i) print_elem(i);
        }
        std::printf("%s\n", oss.str().c_str());
    }
    NWORD_CREATOR_DEFINE_CTX(Dump)
};

// ---- stream 控制 ----

// cuda.sync : --
// 同步 current stream
struct Sync : public NativeWord {
    void run(Stack& /*stack*/) override {
        auto* cctx = dynamic_cast<CudaContext*>(ctx_);
        if (!cctx) vt_fatal_error();
        cctx->sync_current();
    }
    NWORD_CREATOR_DEFINE_CTX(Sync)
};

// cuda.sync_all : --
// 同步所有 stream
struct SyncAll : public NativeWord {
    void run(Stack& /*stack*/) override {
        auto* cctx = dynamic_cast<CudaContext*>(ctx_);
        if (!cctx) vt_fatal_error();
        cctx->sync_all();
    }
    NWORD_CREATOR_DEFINE_CTX(SyncAll)
};

// cuda.set_stream : idx --
// 切换 current stream 到 idx（越界 panic）
struct SetStream : public NativeWord {
    void run(Stack& stack) override {
        auto idx = (size_t)stack.pop_number();
        auto* cctx = dynamic_cast<CudaContext*>(ctx_);
        if (!cctx) vt_fatal_error();
        cctx->set_current(idx);
    }
    NWORD_CREATOR_DEFINE_CTX(SetStream)
};

// ---- event 控制 ----

// cuda.record_event : idx --
// 在 current stream 上 record event[idx]
struct RecordEvent : public NativeWord {
    void run(Stack& stack) override {
        auto idx = (size_t)stack.pop_number();
        auto* cctx = dynamic_cast<CudaContext*>(ctx_);
        if (!cctx) vt_fatal_error();
        cctx->record_event(idx);
    }
    NWORD_CREATOR_DEFINE_CTX(RecordEvent)
};

// cuda.wait_event : idx --
// 让 current stream 等待 event[idx]（不阻塞 host）
struct WaitEvent : public NativeWord {
    void run(Stack& stack) override {
        auto idx = (size_t)stack.pop_number();
        auto* cctx = dynamic_cast<CudaContext*>(ctx_);
        if (!cctx) vt_fatal_error();
        cctx->wait_event(idx);
    }
    NWORD_CREATOR_DEFINE_CTX(WaitEvent)
};

// cuda.sync_event : idx --
// host 阻塞直到 event[idx] 完成
struct SyncEvent : public NativeWord {
    void run(Stack& stack) override {
        auto idx = (size_t)stack.pop_number();
        auto* cctx = dynamic_cast<CudaContext*>(ctx_);
        if (!cctx) vt_fatal_error();
        cctx->sync_event(idx);
    }
    NWORD_CREATOR_DEFINE_CTX(SyncEvent)
};

} // namespace op

Enviroment* create_vt_cuda(int dev) {
    CudaContext* ctx = new CudaContext(dev);
    auto* env = new Enviroment(ctx);

    // common CudaTensor operations
    env->insert_native_word("cuda.create",         op::Create::creator);
    env->insert_native_word("cuda.view",           op::View::creator);
    env->insert_native_word("cuda.load",           op::Load::creator);
    env->insert_native_word("cuda.to_host",        op::ToHost::creator);
    env->insert_native_word("cuda.to_device",      op::ToDevice::creator);
    env->insert_native_word("cuda.zero",           op::Zero::creator);
    env->insert_native_word("cuda.size",           op::Size::creator);
    env->insert_native_word("cuda.is_host",        op::IsHost::creator);
    env->insert_native_word("cuda.is_device",      op::IsDevice::creator);
    env->insert_native_word("cuda.data_ptr",       op::DataPtr::creator);
    env->insert_native_word("cuda.dump",           op::Dump::creator);

    // stream control
    env->insert_native_word("cuda.sync",           op::Sync::creator);
    env->insert_native_word("cuda.sync_all",       op::SyncAll::creator);
    env->insert_native_word("cuda.set_stream",     op::SetStream::creator);

    // event control
    env->insert_native_word("cuda.record_event",   op::RecordEvent::creator);
    env->insert_native_word("cuda.wait_event",     op::WaitEvent::creator);
    env->insert_native_word("cuda.sync_event",     op::SyncEvent::creator);

    return env;
}

} // namespace vt
