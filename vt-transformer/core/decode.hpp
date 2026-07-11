// decode.hpp — DecodeManager
//
// 端口 mini-sglang/python/minisgl/scheduler/decode.py。
// 维护 running_reqs_ 集合(按 uid 排序保证 batch 顺序稳定)。prefill 完成
// 的 req 通过 filter_reqs 进入集合,finished 的 req 通过 remove_req 离开。
// inflight_tokens() 给 PrefillAdder 用作 reserved_size 估计(避免 prefill
// 把 decode 未来要用的 page 抢光)。
//
// 用 shared_ptr<Req>:overlap 模式下,remove 后 Req 可能仍被 batch 引用,
// shared_ptr 让它在所有引用消失前不被析构。

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <vector>

#include "req.hpp"

namespace minisgl {

class DecodeManager {
public:
    explicit DecodeManager(size_t page_size) : page_size_(page_size) {}

    // 把 just_prefilled 合并进 running_reqs_,并丢弃已经不能 decode 的。
    // 对应 mini-sglang decode.py:14-15 的 filter_reqs。
    void filter_reqs(const std::vector<std::shared_ptr<Req>>& just_prefilled) {
        for (const auto& r : just_prefilled) running_reqs_.insert(r);
        for (auto it = running_reqs_.begin(); it != running_reqs_.end();) {
            if (!(*it)->can_decode()) it = running_reqs_.erase(it);
            else                      ++it;
        }
    }

    void remove_req(const std::shared_ptr<Req>& r) { running_reqs_.erase(r); }

    // 按 uid 找;找到返回 shared_ptr 并从 running 中移除,否则空 shared_ptr。
    std::shared_ptr<Req> abort_req(int64_t uid) {
        auto it = std::find_if(running_reqs_.begin(), running_reqs_.end(),
                               [uid](const std::shared_ptr<Req>& r) { return r->uid == uid; });
        if (it == running_reqs_.end()) return {};
        std::shared_ptr<Req> r = *it;
        running_reqs_.erase(it);
        return r;
    }

    // 给 PrefillAdder 用的 reserved_size:running 中所有 req 的 remain_len,
    // 加上每 req 一个 page 的余量(对应 decode.py:28-30 的 inflight_tokens)。
    size_t inflight_tokens() const noexcept {
        size_t tokens_reserved = (page_size_ - 1) * running_reqs_.size();
        size_t sum_remain = 0;
        for (const auto& r : running_reqs_) sum_remain += r->remain_len();
        return sum_remain + tokens_reserved;
    }

    // 把当前 running 拷成一个 decode Batch(已按 uid 排序)。
    // running 为空时返回 nullptr。
    std::unique_ptr<Batch> schedule_next_batch() const {
        if (running_reqs_.empty()) return nullptr;
        auto b = std::make_unique<Batch>();
        b->phase = BatchPhase::Decode;
        b->reqs.reserve(running_reqs_.size());
        for (const auto& r : running_reqs_) b->reqs.push_back(r);
        // running_reqs_ 已按 uid 排序(Cmp),无需再 sort。
        return b;
    }

    bool runnable() const noexcept { return !running_reqs_.empty(); }
    size_t size() const noexcept { return running_reqs_.size(); }

private:
    size_t            page_size_;
    struct Cmp {
        using is_transparent = void;
        bool operator()(const std::shared_ptr<Req>& a,
                        const std::shared_ptr<Req>& b) const noexcept {
            return a->uid < b->uid;
        }
    };
    std::set<std::shared_ptr<Req>, Cmp> running_reqs_;
};

}  // namespace minisgl
