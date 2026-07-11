#ifndef _VT_CUDA_HPP_
#define _VT_CUDA_HPP_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "vt.hpp"

#define COMPLAIN_ERROR_AND_EXIT(what, status) \
    do { \
        fprintf(stderr, "[%s:%d] `%s` returns error: %d.\n", __FILE__, __LINE__, \
                what, status); \
        exit(1); \
    } while (0)

#define CUDA_CHECK(f) \
    do { \
        cudaError_t  s_ = f; \
        if (s_ != cudaSuccess) { \
            fprintf(stderr, "Error string: %s)!\n", cudaGetErrorString(s_)); \
            COMPLAIN_ERROR_AND_EXIT(#f, s_); \
        } \
    } while (0)

#define CUBLAS_CHECK(f) \
    do { \
        cublasStatus_t s_ = f; \
        if (s_ != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cublas error: %s\n", cublasGetStatusString(s_)); \
            COMPLAIN_ERROR_AND_EXIT(#f, (int)s_); \
        } \
    } while (0)

namespace vt {

enum class CudaMemoryType {
    Device,   // cudaMalloc     / cudaFree
    Host,     // cudaMallocHost / cudaFreeHost   (pinned / page-locked)
};

struct CudaTensor : TensorType {
public:
    CudaTensor() = default;

    // Owning allocation: allocates `sz` bytes of the requested memory type.
    CudaTensor(CudaMemoryType type, uint64_t sz) { alloc(type, sz); }

    // Non-owning view: slice [offset, offset+view_bytes) of `src`.
    // `src` is held internally to keep the backing memory alive.
    CudaTensor(tensor_t src, uint64_t offset, uint64_t view_bytes)
        : source_(src) {
        if (!src) vt_panic("CudaTensor view: source is null");
        auto* cuda_src = dynamic_cast<CudaTensor*>(src.get());
        if (!cuda_src) vt_panic("CudaTensor view: source is not a CudaTensor");
        if (offset > cuda_src->size() || view_bytes > cuda_src->size() - offset)
            vt_panic("CudaTensor view: offset+size out of range");
        ptr       = static_cast<char*>(cuda_src->data()) + offset;
        bytes     = view_bytes;
        mem_type_ = cuda_src->mem_type();
        owner     = 0;
    }

    ~CudaTensor() override { release(); }

    CudaTensor(const CudaTensor&)            = delete;
    CudaTensor& operator=(const CudaTensor&) = delete;

    // accessors
    void*           data()      const { return ptr; }
    uint64_t        size()      const { return bytes; }
    bool            is_owner()  const { return owner != 0; }
    bool            is_device() const { return mem_type_ == CudaMemoryType::Device; }
    bool            is_host()   const { return mem_type_ == CudaMemoryType::Host; }
    CudaMemoryType  mem_type()  const { return mem_type_; }

    void dump(std::ostream& os) const override {
        os << "CudaTensor(type=" << (is_device() ? "device" : "host")
           << ", ptr="   << ptr
           << ", bytes=" << bytes
           << ", owner=" << owner
           << ")";
    }

private:
    void*           ptr       = nullptr;
    uint64_t        bytes     = 0;
    int             owner     = 0;
    CudaMemoryType  mem_type_ = CudaMemoryType::Device;
    tensor_t        source_;   // non-null iff this tensor is a view

    void alloc(CudaMemoryType type, uint64_t sz) {
        if (sz == 0) return;
        void* p = nullptr;
        if (type == CudaMemoryType::Device) CUDA_CHECK(cudaMalloc(&p, sz));
        else                                CUDA_CHECK(cudaMallocHost(&p, sz));
        ptr       = p;
        bytes     = sz;
        mem_type_ = type;
        owner     = 1;
    }

    void release() noexcept {
        if (!owner || !ptr) return;
        if (mem_type_ == CudaMemoryType::Device) cudaFree(ptr);
        else                                     cudaFreeHost(ptr);
        ptr   = nullptr;
        bytes = 0;
        owner = 0;
    }
};

struct CudaContext : ComputingContext {
    int              device_id = 0;
    cudaDeviceProp   prop {};
    cublasLtHandle_t cublaslt = nullptr;

    // dev: GPU 设备号；num_streams/num_events: 启动时固定创建的数量。
    explicit CudaContext(int dev = 0, size_t num_streams = 1, size_t num_events = 4)
        : device_id(dev) {
        if (num_streams == 0) num_streams = 1;
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        CUBLAS_CHECK(cublasLtCreate(&cublaslt));
        streams_.reserve(num_streams);
        for (size_t i = 0; i < num_streams; ++i) {
            cudaStream_t s = nullptr;
            CUDA_CHECK(cudaStreamCreate(&s));
            streams_.push_back(s);
        }
        events_.reserve(num_events);
        for (size_t i = 0; i < num_events; ++i) {
            cudaEvent_t e = nullptr;
            CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            events_.push_back(e);
        }
    }

    ~CudaContext() override {
        for (auto s : streams_) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
        }
        for (auto e : events_) {
            cudaEventDestroy(e);
        }
        if (cublaslt) cublasLtDestroy(cublaslt);
    }

    CudaContext(const CudaContext&)            = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    // ---- 设备属性便捷访问 ----
    int    sm_version()       const { return prop.major * 10 + prop.minor; }  // e.g. 90 for H100
    int    num_sms()          const { return prop.multiProcessorCount; }
    size_t total_global_mem() const { return prop.totalGlobalMem; }
    size_t shared_mem_per_block() const { return prop.sharedMemPerBlock; }
    int    warp_size()        const { return prop.warpSize; }   // 32 on all NVIDIA GPUs

    // 访问
    cudaStream_t get_stream(size_t idx) const {
        if (idx >= streams_.size())
            vt_panic("CudaContext::get_stream: index out of range");
        return streams_[idx];
    }

    cudaStream_t current_stream() const { return streams_[current_]; }
    size_t       current() const        { return current_; }
    void         set_current(size_t idx) {
        if (idx >= streams_.size())
            vt_panic("CudaContext::set_current: index out of range");
        current_ = idx;
    }

    size_t num_streams() const { return streams_.size(); }

    // 同步
    void sync(size_t idx) const {
        CUDA_CHECK(cudaStreamSynchronize(get_stream(idx)));
    }
    void sync_current() const {
        CUDA_CHECK(cudaStreamSynchronize(current_stream()));
    }
    void sync_all() const {
        for (auto s : streams_) CUDA_CHECK(cudaStreamSynchronize(s));
    }

    // ---- events ----
    // 所有 event 在构造时一次性创建，用 cudaEventDisableTiming 标志（纯同步用途，比默认快）。
    cudaEvent_t get_event(size_t idx) const {
        if (idx >= events_.size())
            vt_panic("CudaContext::get_event: index out of range");
        return events_[idx];
    }

    size_t num_events() const { return events_.size(); }

    // 在 current stream 上 record event[idx]
    void record_event(size_t idx) {
        CUDA_CHECK(cudaEventRecord(get_event(idx), current_stream()));
    }

    // 让 current stream 等待 event[idx]（不阻塞 host；如果 event 还没被 record，
    // 立即通过；否则等 record 它的那条 stream 走到 record 点）。
    void wait_event(size_t idx) {
        CUDA_CHECK(cudaStreamWaitEvent(current_stream(), get_event(idx), 0));
    }

    // host 阻塞直到 event[idx] 完成（即 record 它的 stream 跑完 record 之前的所有工作）。
    void sync_event(size_t idx) const {
        CUDA_CHECK(cudaEventSynchronize(get_event(idx)));
    }

private:
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t>  events_;
    size_t                    current_ = 0;
};

Enviroment* create_vt_cuda(int dev = 0);

} // namespace

#endif