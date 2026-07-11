// pinned_allocator.hpp — std::vector 用的 pinned host memory allocator
//
// 让 ForwardOutput::next_tokens_cpu 自带 pinned storage,cudaMemcpyAsync(host, ...)
// 才是真异步(对非 pinned host 内存,cudaMemcpyAsync 会隐式同步)。
//
// MINISGL_USE_CUDA on  -> cudaMallocHost / cudaFreeHost
// MINISGL_USE_CUDA off -> 退化成 malloc / free(链接无需 -lcudart)
// Mock 路径 / schedule_test 不定义 MINISGL_USE_CUDA,所以零成本退化为普通堆内存。

#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

#ifdef MINISGL_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace minisgl {

template <typename T>
struct PinnedAllocator {
    using value_type = T;

    PinnedAllocator() noexcept = default;
    template <typename U>
    PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > static_cast<std::size_t>(-1) / sizeof(T)) {
            throw std::bad_alloc();
        }
#ifdef MINISGL_USE_CUDA
        void* p = nullptr;
        if (cudaMallocHost(&p, n * sizeof(T)) != cudaSuccess) {
            throw std::bad_alloc();
        }
#else
        void* p = std::malloc(n * sizeof(T));
        if (!p) throw std::bad_alloc();
#endif
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t /*n*/) noexcept {
        if (!p) return;
#ifdef MINISGL_USE_CUDA
        cudaFreeHost(p);
#else
        std::free(p);
#endif
    }
};

template <typename T, typename U>
constexpr bool operator==(const PinnedAllocator<T>&, const PinnedAllocator<U>&) noexcept {
    return true;
}
template <typename T, typename U>
constexpr bool operator!=(const PinnedAllocator<T>&, const PinnedAllocator<U>&) noexcept {
    return false;
}

}  // namespace minisgl
