#include "vt_cuda.hpp"

#include <cstring>

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

// ---- stream 控制 ----

// cuda.sync : --
// 同步 current stream
struct Sync : public NativeWord {
    void run(Stack& stack) override {
        auto* cctx = dynamic_cast<CudaContext*>(ctx_);
        if (!cctx) vt_fatal_error();
        cctx->sync_current();
    }
    NWORD_CREATOR_DEFINE_CTX(Sync)
};

// cuda.sync_all : --
// 同步所有 stream
struct SyncAll : public NativeWord {
    void run(Stack& stack) override {
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
    env->insert_native_word("cuda.to_host",        op::ToHost::creator);
    env->insert_native_word("cuda.to_device",      op::ToDevice::creator);
    env->insert_native_word("cuda.zero",           op::Zero::creator);
    env->insert_native_word("cuda.size",           op::Size::creator);
    env->insert_native_word("cuda.is_host",        op::IsHost::creator);
    env->insert_native_word("cuda.is_device",      op::IsDevice::creator);

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
