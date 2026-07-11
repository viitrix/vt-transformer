// cache_manager.hpp — Page-pool owner that wraps RadixCache.
//
// Mirrors python/minisgl/scheduler/cache.py:CacheManager:
//   * Owns the free_slots page pool (page-aligned token offsets).
//   * Drives RadixCache for match/insert/lock (passthroughs).
//   * allocate(n_pages) pops from free_slots, evicting from the tree when
//     the pool runs dry.
//   * free(token_indices) returns request-private pages to the pool — used
//     for orphan pages (allocated but rejected by insert_prefix) and the
//     sub-page tail when a request finishes.
//   * check_integrity enforces page conservation:
//       free_slots.size() + cache.total_size / page_size == num_pages
//
// Two invariants kept at all times:
//   1. Every entry in free_slots_ is page-aligned (offset % page_size == 0).
//   2. allocated/evicted indices flow back through free_slots_; the tree
//      never sees the pool, and the pool never sees the tree.
//
// What this file does NOT do (out of scope for the mock scheduler):
//   * Per-request page_table bookkeeping — caller builds page_ids inline.
//   * lazy_free_region batching — no concurrent batches in mock mode.
//   * cache_req lifecycle helper — single-threaded mock has no orphans.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "radix.hpp"

namespace minisgl {

template <typename TokenT,
          typename IndexT   = int32_t,
          template <typename> class Allocator = HostAllocator>
class CacheManager {
public:
    using RadixCacheT  = RadixCache<TokenT, IndexT, Allocator>;
    using CacheHandle  = typename RadixCacheT::CacheHandle;
    using MatchResult  = typename RadixCacheT::MatchResult;
    using InsertResult = typename RadixCacheT::InsertResult;
    using SizeInfo     = typename RadixCacheT::SizeInfo;
    using IndexBuffer  = typename RadixCacheT::IndexBuffer;

    CacheManager(size_t page_size, size_t num_pages)
        : page_size_(page_size == 0 ? 1 : page_size),
          num_pages_(num_pages),
          cache_(page_size_) {
        free_slots_.reserve(num_pages_);
        for (size_t i = 0; i < num_pages_; ++i) {
            free_slots_.push_back(static_cast<IndexT>(i * page_size_));
        }
    }

    CacheManager(const CacheManager&)            = delete;
    CacheManager& operator=(const CacheManager&) = delete;
    CacheManager(CacheManager&&) noexcept            = default;
    CacheManager& operator=(CacheManager&&) noexcept = default;

    // ---------- passthroughs to RadixCache ----------
    MatchResult match_prefix(const std::vector<TokenT>& input_ids) {
        return cache_.match_prefix(input_ids);
    }

    InsertResult insert_prefix(const std::vector<TokenT>& input_ids,
                               const std::vector<IndexT>& indices) {
        return cache_.insert_prefix(input_ids, indices);
    }

    void lock_handle(const CacheHandle& h, bool unlock = false) {
        cache_.lock_handle(h, unlock);
    }

    IndexBuffer get_matched_indices(const CacheHandle& h) const {
        return cache_.get_matched_indices(h);
    }

    SizeInfo size_info() const noexcept { return cache_.size_info(); }

    // ---------- accessors ----------
    size_t page_size() const noexcept     { return page_size_; }
    size_t num_pages() const noexcept     { return num_pages_; }
    size_t num_free_pages() const noexcept { return free_slots_.size(); }

    // Total room for new requests, in tokens. Admission control checks this
    // against the request's page-aligned need.
    size_t available_size() const noexcept {
        return cache_.size_info().evictable_size + free_slots_.size() * page_size_;
    }

    // ---------- page pool ----------
    // Returns page-aligned token offsets (length == needed_pages). Evicts
    // from the tree if free_slots runs dry. Throws if evictable_size is
    // also insufficient (caller must check available_size() first).
    std::vector<IndexT> allocate(size_t needed_pages) {
        if (needed_pages > free_slots_.size()) {
            const size_t shortage_pages = needed_pages - free_slots_.size();
            // evict() takes token count, returns per-token indices.
            IndexBuffer evicted = cache_.evict(shortage_pages * page_size_);
            const std::vector<IndexT> v = evicted.to_host();
            // Downsample per-token -> per-page (one entry per page_size_ tokens).
            for (size_t i = 0; i < v.size(); i += page_size_) {
                free_slots_.push_back(v[i]);
            }
        }
        std::vector<IndexT> out;
        out.reserve(needed_pages);
        for (size_t i = 0; i < needed_pages; ++i) {
            out.push_back(free_slots_.back());
            free_slots_.pop_back();
        }
        return out;
    }

    // Returns request-private pages to the pool. Input is per-token indices
    // (matching RadixCache's value-array layout); internally downsampled
    // to per-page before pushing to free_slots_.
    void free(const std::vector<IndexT>& token_indices) {
        for (size_t i = 0; i < token_indices.size(); i += page_size_) {
            free_slots_.push_back(token_indices[i]);
        }
    }

    // ---------- integrity ----------
    void check_integrity() const {
        cache_.check_integrity();
        const size_t cache_pages = cache_.size_info().total_size() / page_size_;
        if (free_slots_.size() + cache_pages != num_pages_) {
            throw std::runtime_error(
                "CacheManager integrity: page count leak (free=" +
                std::to_string(free_slots_.size()) + ", cache=" +
                std::to_string(cache_pages) + ", total=" +
                std::to_string(num_pages_) + ")");
        }
    }

    // For debugging / scheduler logs.
    RadixCacheT&       cache()       { return cache_; }
    const RadixCacheT& cache() const { return cache_; }

private:
    size_t              page_size_;
    size_t              num_pages_;
    RadixCacheT         cache_;
    std::vector<IndexT> free_slots_;
};

}  // namespace minisgl
