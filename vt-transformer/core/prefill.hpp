// prefill.hpp — PrefillManager + PrefillAdder(无 chunked)
//
// 端口 mini-sglang/python/minisgl/scheduler/prefill.py 的 PrefillManager 和
// PrefillAdder。chunked prefill 首版砍掉:结构上 Req.chunked 字段保留,
// 但 PrefillAdder 不切分。后续要做 chunked 时主要改 try_add_one。
//
// 准入逻辑(对应 prefill.py:39-113 _try_allocate_one + _add_one_req):
//   1. token_budget 还有?
//   2. table_manager 还有 slot?
//   3. match_prefix(input_ids[:-1]) 拿 cached_len
//   4. estimated = extend + output_len,够覆盖 available_size 吗?
//   5. lock handle,二次确认(锁后 evictable 可能变小);不够就 unlock 退回
//   6. allocate table slot,把 cached 部分的 page_indices 写进 Req
//   7. 构造 Req 并 emplace 进 Scheduler 的 live_reqs_(shared_ptr)

#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cache.hpp"
#include "decode.hpp"
#include "req.hpp"
#include "table.hpp"

namespace minisgl {

class PrefillAdder {
public:
    PrefillAdder(size_t token_budget,
                 size_t reserved_size,
                 CacheManager<int32_t, int32_t, HostAllocator>& cm,
                 TableManager& tm,
                 std::unordered_map<int64_t, std::shared_ptr<Req>>& live_reqs)
        : token_budget_(token_budget),
          reserved_size_(reserved_size),
          cm_(cm),
          tm_(tm),
          live_reqs_(live_reqs) {}

    // 成功返回 live_reqs_ 里的 shared_ptr(同时也在返回的 batch 中持有);
    // 失败(预算/容量不足)返回空 shared_ptr。
    // 失败时不会留下任何副作用(handle 没锁、slot 没占)。
    std::shared_ptr<Req> try_add_one(const PendingReq& pending) {
        if (token_budget_ == 0) return {};
        if (tm_.available_size() == 0) return {};

        // 首版不支持 chunked:extend_len 超出 token_budget 直接拒(对应
        // python 里 chunk_size < remain_len 才走 chunked 分支,我们直接 break)。
        // 这检查放在前面,避免无谓的 match/lock。
        // (cached_len 来自下面 match,这里先用 input_len 估上界。)
        const size_t extend_upper = pending.input_len();  // worst case: cached_len=0
        if (extend_upper > token_budget_) return {};

        // match_prefix 用 input_ids[:-1]:跟 mini-sglang cache.py:30 一致,
        // 不算最后一个 token(因为最后一个 token 没有 KV,不该被匹配)。
        std::vector<int32_t> probe;
        if (pending.input_ids.size() > 1) {
            probe.assign(pending.input_ids.begin(), pending.input_ids.end() - 1);
        }
        auto match = cm_.match_prefix(probe);
        size_t cached_len = match.handle.cached_len;

        size_t extend_len_v = pending.input_len() - cached_len;
        size_t estimated_len = extend_len_v + pending.output_len();

        if (estimated_len + reserved_size_ > cm_.available_size()) return {};
        cm_.lock_handle(match.handle);
        if (estimated_len + reserved_size_ > cm_.available_size()) {
            cm_.lock_handle(match.handle, /*unlock=*/true);
            return {};
        }

        int64_t table_idx = tm_.allocate();

        // 构造 Req 并填充初始字段。
        auto req = std::make_shared<Req>();
        req->uid            = pending.uid;
        req->table_idx      = table_idx;
        req->cached_len     = cached_len;
        req->sampling_params = pending.sampling_params;
        req->input_ids.assign(pending.input_ids.begin(), pending.input_ids.end());
        req->max_device_len = pending.input_len() + pending.output_len();
        req->device_len     = pending.input_len();
        req->cache_handle   = match.handle;
        req->page_indices.assign(req->max_device_len, -1);

        // 把 cached 部分的 page_indices 写入(从 matched handle 取)。
        if (cached_len > 0) {
            std::vector<int32_t> matched_idx = cm_.get_matched_indices(match.handle).to_host();
            for (size_t i = 0; i < cached_len && i < matched_idx.size(); ++i) {
                req->page_indices[i] = matched_idx[i];
            }
        }

        // 接进 live_reqs_,所有权转移(shared_ptr)。
        live_reqs_.emplace(req->uid, req);

        // chunked 首版不做:超出 budget 直接拒(由调用者在循环里 break)。
        // 这里成功 admit 一个完整 req,扣掉它的 extend_len。
        token_budget_  -= extend_len_v;
        reserved_size_ += pending.output_len() + extend_len_v;

        return req;
    }

    size_t token_budget() const noexcept { return token_budget_; }

private:
    size_t           token_budget_;
    size_t           reserved_size_;
    CacheManager<int32_t, int32_t, HostAllocator>& cm_;
    TableManager&    tm_;
    std::unordered_map<int64_t, std::shared_ptr<Req>>& live_reqs_;
};

class PrefillManager {
public:
    PrefillManager(CacheManager<int32_t, int32_t, HostAllocator>& cm,
                   TableManager& tm,
                   DecodeManager& dm,
                   std::unordered_map<int64_t, std::shared_ptr<Req>>& live_reqs,
                   size_t max_seq_len)
        : cm_(cm), tm_(tm), dm_(dm), live_reqs_(live_reqs),
          max_seq_len_(max_seq_len) {}

    // 端口 scheduler.py:177-189 的 max_seq_len 守门:超长 input 直接丢,
    // 否则把 max_tokens clamp 到 max_seq_len - input_len。已经符合的请求原样入队。
    void add_one_req(const UserMsg& msg) {
        const size_t input_len = msg.input_ids.size();
        if (input_len > max_seq_len_) {
            std::cerr << "[prefill] drop uid=" << msg.uid
                      << ": input_len=" << input_len
                      << " > max_seq_len=" << max_seq_len_ << "\n";
            return;
        }
        const size_t max_output_len = max_seq_len_ - input_len;
        PendingReq p;
        p.uid             = msg.uid;
        p.input_ids       = msg.input_ids;
        p.sampling_params = msg.sampling_params;
        if (p.sampling_params.max_tokens > static_cast<int64_t>(max_output_len)) {
            std::cerr << "[prefill] clamp uid=" << msg.uid << ": max_tokens "
                      << p.sampling_params.max_tokens << " -> " << max_output_len
                      << " (max_seq_len=" << max_seq_len_ << ")\n";
            p.sampling_params.max_tokens = static_cast<int64_t>(max_output_len);
        }
        pending_list_.push_back(std::move(p));
    }

    // 找不到可跑的 req 返回 nullptr。
    std::unique_ptr<Batch> schedule_next_batch(size_t prefill_budget) {
        if (pending_list_.empty()) return nullptr;

        PrefillAdder adder(prefill_budget, dm_.inflight_tokens(), cm_, tm_, live_reqs_);
        std::vector<std::shared_ptr<Req>> admitted;
        size_t consumed = 0;
        while (consumed < pending_list_.size()) {
            auto r = adder.try_add_one(pending_list_[consumed]);
            if (!r) break;
            admitted.push_back(r);
            ++consumed;
        }
        if (admitted.empty()) return nullptr;

        // 丢弃已 consumed 的 pending;剩下的留在 pending_list_ 头部。
        pending_list_.erase(pending_list_.begin(),
                            pending_list_.begin() + static_cast<ptrdiff_t>(consumed));

        auto b = std::make_unique<Batch>();
        b->phase = BatchPhase::Prefill;
        b->reqs  = std::move(admitted);
        return b;
    }

    // abort:从 pending 找 uid。pending 里的 PendingReq 还没分资源,直接删。
    // 返回空 shared_ptr(语义上 Scheduler 不需要为 pending abort 做 free)。
    std::shared_ptr<Req> abort_req(int64_t uid) {
        for (auto it = pending_list_.begin(); it != pending_list_.end(); ++it) {
            if (it->uid == uid) {
                pending_list_.erase(it);
                return {};
            }
        }
        return {};
    }

    bool runnable() const noexcept { return !pending_list_.empty(); }
    size_t pending_size() const noexcept { return pending_list_.size(); }

private:
    CacheManager<int32_t, int32_t, HostAllocator>& cm_;
    TableManager&    tm_;
    DecodeManager&   dm_;
    std::unordered_map<int64_t, std::shared_ptr<Req>>& live_reqs_;
    const size_t     max_seq_len_;
    std::vector<PendingReq> pending_list_;
};

}  // namespace minisgl
