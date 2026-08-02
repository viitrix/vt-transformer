// vt_scheduler.cpp — FullScheduler 实现（推理计算核心调度器）
//
// 详见 vt_scheduler.hpp 顶部注释中的 Continue Batch 模型与职责切分。
// 文件末尾对 default 类型 `<int32_t, int32_t>` 做 explicit instantiation；
// 其它类型需要在使用方自行 instantiate。

#include "vt_scheduler.hpp"

#include <algorithm>
#include <utility>

namespace vt {

// ============================================================================
// 构造
// ============================================================================

template <typename TokenT, typename IndexT>
FullScheduler<TokenT, IndexT>::FullScheduler(Config config, Engine* engine)
    : config_(config),
      engine_(engine),
      cache_(std::make_unique<Cache>(config.max_running_reqs,
                                       config.num_pages,
                                       config.max_seq_len,
                                       config.page_size)) {
    vt_assert(engine != nullptr, "FullScheduler: engine must not be null");
    vt_assert(config.max_running_reqs   > 0, "FullScheduler: max_running_reqs must be > 0");
    vt_assert(config.num_pages          > 0, "FullScheduler: num_pages must be > 0");
    vt_assert(config.max_seq_len        > 0, "FullScheduler: max_seq_len must be > 0");
    vt_assert(config.page_size          > 0, "FullScheduler: page_size must be > 0");
    vt_assert(config.max_extend_tokens  > 0, "FullScheduler: max_extend_tokens must be > 0");
    vt_assert(config.default_max_output > 0, "FullScheduler: default_max_output must be > 0");
}

// ============================================================================
// 外部 IO 入口
// ============================================================================

template <typename TokenT, typename IndexT>
void FullScheduler<TokenT, IndexT>::add_req(Req req) {
    vt_assert(req.state == Req::Waiting, "FullScheduler::add_req: state must be Waiting");
    vt_assert(!req.input.empty(), "FullScheduler::add_req: input must be non-empty");
    pending_.push_back(std::move(req));
}

template <typename TokenT, typename IndexT>
bool FullScheduler<TokenT, IndexT>::abort_req(uint64_t uid) {
    // 1) 还在 pending_（未占用资源）→ 直接 erase
    auto pit = find_in_pending(uid);
    if (pit != pending_.end()) {
        pending_.erase(pit);
        return true;
    }
    // 2) 已在 running_（已分配 row + radix lock）→ 归还资源再 erase
    auto rit = find_in_running(uid);
    if (rit != running_.end()) {
        Req& r = *rit;

        // overlap 模式下，req 可能正在 inflight_batch_ 里（GPU 还在算）。
        // 此时不能立刻 erase——inflight_batch_ 还持着 Req*，下一步 phase 1 要用。
        // 标记为 Finished，让下一次 step_overlap 的 phase 1 走 abort 分支：
        // 跳过 record_predicted，直接 free_finished 归还资源。
        if (has_inflight() && is_inflight_req_(&r)) {
            r.finish();
            return true;
        }

        // 拼 tokens：input[0..prefill_pos) + output[0..decode_pos)
        std::vector<Token> tokens;
        tokens.reserve((size_t)r.cached_len());
        tokens.insert(tokens.end(), r.input.begin(),
                                    r.input.begin() + r.prefill_pos);
        tokens.insert(tokens.end(), r.output.begin(),
                                    r.output.begin() + r.decode_pos);

        typename Cache::FinishInput fi{
            /*table_idx*/       r.table_idx,
            /*prepare_result*/  {r.node, r.radix_hit_len},
            /*cur_cached_len*/  r.cached_len(),
            /*tokens*/          std::move(tokens),
        };
        cache_->finished(fi);

        running_.erase(rit);
        return true;
    }
    return false;
}

// ============================================================================
// 调度循环
// ============================================================================

template <typename TokenT, typename IndexT>
typename FullScheduler<TokenT, IndexT>::StepResult
FullScheduler<TokenT, IndexT>::step() {
    StepResult ret{/*ran_batch=*/false, /*finished_count=*/0, /*results=*/{}};
    if (!has_work()) return ret;

    cur_batch_.reqs.clear();

    // 选 batch：prefill 优先（对齐 mini-sglang scheduler.py:219-225）
    int n_prefill = try_schedule_prefill();
    if (n_prefill > 0) {
        cur_batch_.phase = BatchT::Prefill;
    } else if (try_schedule_decode()) {
        cur_batch_.phase = BatchT::Decode;
    } else {
        return ret;  // 没选出 batch（比如容量不足）
    }

    if (!prepare_batch()) {
        // 资源不够：本步空转，req 留在原 list 等下一步。
        // prepare_batch 失败时已把已分配的资源回滚。
        cur_batch_.reqs.clear();
        return ret;
    }

    Output out = engine_->forward(cur_batch_, *cache_->pt());

    std::vector<int> finished_idx;
    process_results(out, finished_idx);

    // 收 results：process_results 已经把预测 token 追加到 req.output，
    // finished_idx 也已收集好；此时 req 还在 running_，引用仍然有效。
    // free_finished 之后被 erase 的 Req& 就成悬挂了，所以必须在它之前读。
    ret.results.reserve(cur_batch_.size());
    for (int i = 0; i < cur_batch_.size(); ++i) {
        bool fin = std::find(finished_idx.begin(), finished_idx.end(), i)
                   != finished_idx.end();
        ret.results.push_back({cur_batch_.reqs[i]->id, out.next_tokens[i], fin});
    }

    free_finished(finished_idx);

    ret.ran_batch      = true;
    ret.finished_count = (int)finished_idx.size();

    cur_batch_.reqs.clear();
    return ret;
}

// ============================================================================
// Overlap 调度循环
// ============================================================================

template <typename TokenT, typename IndexT>
typename FullScheduler<TokenT, IndexT>::StepResult
FullScheduler<TokenT, IndexT>::step_overlap() {
    StepResult ret{/*ran_batch=*/false, /*finished_count=*/0, /*results=*/{}};

    // === 阶段 1: 处理上次 in-flight 的 forward ===
    if (!inflight_batch_.reqs.empty()) {
        Output prev_out = engine_->wait(std::move(inflight_handle_));
        // 把 inflight 换到 cur_batch_，复用 process_inflight_results / free_finished
        cur_batch_ = inflight_batch_;
        inflight_batch_.reqs.clear();

        std::vector<int> finished_idx;
        process_inflight_results(prev_out, finished_idx);

        // 收 results：跟 sync step() 一样在 free_finished 之前读，避免悬挂。
        // next_token 直接取 engine 产出（inflight 期间被 abort 的 req 跳过了
        // record_predicted，但 engine 还是产了 token——这里报它实际产出的值，
        // 与 mini-sglang _process_last_data 中 next_token = next_tokens_cpu[i] 一致）。
        ret.results.reserve(cur_batch_.size());
        for (int i = 0; i < cur_batch_.size(); ++i) {
            bool fin = std::find(finished_idx.begin(), finished_idx.end(), i)
                       != finished_idx.end();
            ret.results.push_back({cur_batch_.reqs[i]->id, prev_out.next_tokens[i], fin});
        }

        free_finished(finished_idx);

        ret.finished_count = (int)finished_idx.size();
        cur_batch_.reqs.clear();
    }

    // === 阶段 2: 选下一个 batch（prefill 优先，跟 sync 一致）===
    int n_prefill = try_schedule_prefill();
    if (n_prefill > 0) {
        cur_batch_.phase = BatchT::Prefill;
    } else if (try_schedule_decode()) {
        cur_batch_.phase = BatchT::Decode;
    } else {
        return ret;  // 没选出 batch
    }

    if (!prepare_batch()) {
        cur_batch_.reqs.clear();
        return ret;
    }

    // === 阶段 3: speculative commit ===
    // 把刚 submit 的 forward 会写的 KV 计入 cached_len，这样下一步的 prepare_batch
    // 算 slot 时能拿到正确的 cached_len（否则同一 req 连续两步 decode 会撞 slot）。
    //   prefill：commit_prefill_kv —— 状态仍是 Prefill（由下步 process 时 record_predicted 切到 Decode）
    //   decode ：commit_decode_kv —— 状态仍是 Decode
    if (cur_batch_.is_prefill()) {
        for (Req* r : cur_batch_.reqs) r->commit_prefill_kv();
    } else {
        for (Req* r : cur_batch_.reqs) r->commit_decode_kv();
    }

    // === 阶段 4: 提交 forward_async（默认实现就是同步包一层；真实 engine 重写做 async）===
    inflight_handle_ = engine_->forward_async(cur_batch_, *cache_->pt());
    inflight_batch_  = cur_batch_;  // 拷贝 Req* 指针 + phase；reqs 非空 ⇒ has_inflight()==true

    // prefill reqs 立刻从 pending_ 搬到 running_——让 abort_req 在 inflight 期间
    // 也能经由 running_ 分支找到它们（否则 pending_ 分支会盲 erase，留下悬挂指针）。
    // decode reqs 已经在 running_ 里，不动。
    // 注意：state 仍是 Prefill，要等下一步 phase 1 record_predicted 才切到 Decode。
    if (cur_batch_.is_prefill()) {
        for (Req* r : cur_batch_.reqs) {
            auto it = find_in_pending(r->id);
            vt_assert(it != pending_.end(),
                      "FullScheduler::step_overlap: prefill req missing in pending_");
            running_.splice(running_.end(), pending_, it);
        }
    }

    cur_batch_.reqs.clear();

    ret.ran_batch = true;
    return ret;
}

template <typename TokenT, typename IndexT>
typename FullScheduler<TokenT, IndexT>::ReqIter
FullScheduler<TokenT, IndexT>::find_in_pending(uint64_t uid) {
    for (auto it = pending_.begin(); it != pending_.end(); ++it) {
        if (it->id == uid) return it;
    }
    return pending_.end();
}

template <typename TokenT, typename IndexT>
typename FullScheduler<TokenT, IndexT>::ReqIter
FullScheduler<TokenT, IndexT>::find_in_running(uint64_t uid) {
    for (auto it = running_.begin(); it != running_.end(); ++it) {
        if (it->id == uid) return it;
    }
    return running_.end();
}

// ============================================================================
// 内部：选 batch
// ============================================================================

template <typename TokenT, typename IndexT>
int FullScheduler<TokenT, IndexT>::try_schedule_prefill() {
    if (pending_.empty()) return 0;

    // running_ 在飞 token 估算（对齐 mini-sglang prefill.py:131-134 PrefillAdder）：
    //   - 每个 running req 预留 1 page 的 head-room（防 decode 触底）
    //   - 加上它剩余的 output budget
    const int page_size       = config_.page_size;
    int       reserved_tokens = 0;
    for (const Req& r : running_) {
        int remain_output = config_.default_max_output - r.decode_pos;
        reserved_tokens += (page_size - 1) + std::max(0, remain_output);
    }

    int token_budget = config_.max_extend_tokens;
    int n_selected   = 0;

    for (auto it = pending_.begin(); it != pending_.end(); ++it) {
        Req& r = *it;
        // 防御性过滤：pending_ 正常只含 state=Waiting 的 req——
        // overlap 模式下 inflight prefill 在 submit 时已搬到 running_（step_overlap 阶段 4），
        // sync 模式下 process_results 末尾才搬。两种模式下中间窗口对调用方都不可见，
        // 但保留这道 check 以防未来某个分支打破不变式时不会 silently 选错。
        if (r.state != Req::Waiting) continue;
        if (token_budget <= 0) break;

        // 估算：本次 prefill 的 input token 数 + 后续 decode 总预算。
        // 本实现暂不支持单 req 跨 step 的 chunked prefill，整段 input 一次性入 batch。
        int extend_len = (int)r.input.size();
        int estimated  = extend_len + config_.default_max_output + reserved_tokens;

        int avail = (int)cache_->available_size();

        // 容量瓶颈分两种情形：
        //   (a) 仅"running head-room + 一条 req 的最小未来开销"就超容量：
        //       即使后面遇到 extend_len=0 的 req 也救不回来——直接停。
        if (config_.default_max_output + reserved_tokens > avail) break;

        //   (b) 本 req 的 input 太长把自己撑爆，但容量其实还有富余：
        //       跳过它，后面更短的 req 可能还能装下。
        if (estimated > avail) continue;

        // 单 req 比本次 token budget 还大：先跳过它（不阻塞后面更小的 req）。
        // 简单策略；mini-sglang 这里会做 chunked prefill，本实现 defer。
        if (extend_len > token_budget) continue;

        cur_batch_.reqs.push_back(&r);
        token_budget     -= extend_len;
        reserved_tokens  += extend_len + config_.default_max_output;
        ++n_selected;
    }
    return n_selected;
}

template <typename TokenT, typename IndexT>
bool FullScheduler<TokenT, IndexT>::try_schedule_decode() {
    if (running_.empty()) return false;
    for (Req& r : running_) {
        cur_batch_.reqs.push_back(&r);
    }
    return true;
}

// ============================================================================
// 内部：准备资源
// ============================================================================

template <typename TokenT, typename IndexT>
bool FullScheduler<TokenT, IndexT>::prepare_batch() {
    // 失败语义：原子——任何一步失败，前面已动的资源全部回滚到进入前的状态。
    //
    // prefill 与 decode 走不同路径：
    //   prefill：每个 req 都是新入选——alloc_row + CacheManager.prepare（radix
    //            match+lock + 写命中段），失败时反向 finished（unlock + free_row）。
    //   decode ：req 已在 running_，table_idx / radix lock 都还在，**不重新 prepare**——
    //            只需给本步新写的 slot 申请 page。
    //
    // 共同的尾巴：allocate_pages。prefill 失败要把 prepare 也一起反向回滚；
    //            decode 失败什么都不用回滚（page 池原子分配，失败时啥也没动）。

    std::vector<typename Cache::AllocItem> alloc_items;
    alloc_items.reserve(cur_batch_.reqs.size());

    if (cur_batch_.is_prefill()) {
        // (a) per-req prepare
        int n_prepared = 0;
        for (Req* r : cur_batch_.reqs) {
            Index row = cache_->pt()->alloc_row();
            if (row == PageTable<Index>::kInvalid) {
                // 行池不够：回滚已 prepare 的 req
                for (int j = 0; j < n_prepared; ++j) {
                    rollback_prepare_(cur_batch_.reqs[j]);
                }
                return false;
            }
            r->table_idx = row;

            auto pr = cache_->prepare(row, r->input);
            r->node          = pr.node;
            r->prefill_pos   = pr.cached_len;
            r->radix_hit_len = pr.cached_len;
            r->to_prefill();
            ++n_prepared;

            alloc_items.push_back({row, pr.cached_len, (int)r->input.size()});
        }
    } else {
        // decode：每个 req 已经在 running_ 里，直接算新 slot 范围
        for (Req* r : cur_batch_.reqs) {
            int cur_cached = r->prefill_pos + r->decode_pos;
            alloc_items.push_back({r->table_idx, cur_cached, cur_cached + 1});
        }
    }

    // (b) 一次性分配 page（page 池原子分配，失败时 pt_ 内部不动）
    if (!cache_->allocate_pages(alloc_items)) {
        if (cur_batch_.is_prefill()) {
            for (Req* done : cur_batch_.reqs) {
                rollback_prepare_(done);
            }
        }
        return false;
    }
    return true;
}

template <typename TokenT, typename IndexT>
bool FullScheduler<TokenT, IndexT>::is_inflight_req_(const Req* r) const {
    if (inflight_batch_.reqs.empty()) return false;
    for (const Req* ir : inflight_batch_.reqs) {
        if (ir == r) return true;
    }
    return false;
}

template <typename TokenT, typename IndexT>
void FullScheduler<TokenT, IndexT>::rollback_prepare_(Req* r) {
    // 用空 tokens + cur_cached_len=0 调 finished：内部不插入 radix，仅 unlock +
    // free_row（命中段的 page_id 由 radix 持有，不能 free）。
    typename Cache::FinishInput empty{
        /*table_idx*/      r->table_idx,
        /*prepare_result*/ {r->node, r->radix_hit_len},
        /*cur_cached_len*/ 0,
        /*tokens*/         {},
    };
    cache_->finished(empty);

    r->table_idx     = (Index)-1;
    r->node          = nullptr;
    r->radix_hit_len = 0;
    r->prefill_pos   = 0;
    r->state         = Req::Waiting;
}

// ============================================================================
// 内部：处理 forward 结果
// ============================================================================

template <typename TokenT, typename IndexT>
void FullScheduler<TokenT, IndexT>::process_results(const Output& out,
                                                  std::vector<int>& finished_idx) {
    vt_assert((int)out.next_tokens.size() == cur_batch_.size(),
              "FullScheduler::process_results: next_tokens size must == batch size");

    const Token eos     = engine_->eos_token_id();
    const int   max_out = config_.default_max_output;

    for (int i = 0; i < cur_batch_.size(); ++i) {
        Req&   r   = *cur_batch_.reqs[i];
        Token  tok = out.next_tokens[i];

        // commit + record：把刚 submit 的 forward 写的 KV 计入 cached_len，
        // 把它预测的 token 追加到 output。on_prefill_done 还会把状态 Prefill→Decode。
        if (cur_batch_.is_prefill()) {
            r.on_prefill_done(tok);
        } else {
            r.on_decode_done(tok);
        }

        // EOS / max_output 触发 finished。output.size() == 已预测的 token 数；
        // 达到上限说明本步产出的 token 是最后一个。
        bool hit_eos = (eos != Token(-1)) && (tok == eos);
        bool hit_max = (int)r.output.size() >= max_out;

        if (hit_eos || hit_max) {
            r.finish();
            finished_idx.push_back(i);
        }
    }

    // prefill 后所有 req 都从 pending_ 转入 running_——
    // finished 的也走，free_finished 会立刻从 running_ 把它们 erase 掉。
    // 这样 free_finished 只需要在一个地方找 req。
    if (cur_batch_.is_prefill()) {
        for (int i = 0; i < cur_batch_.size(); ++i) {
            Req& r = *cur_batch_.reqs[i];
            auto it = find_in_pending(r.id);
            vt_assert(it != pending_.end(),
                      "FullScheduler::process_results: prefill req missing in pending_");
            running_.splice(running_.end(), pending_, it);
        }
    }
}

template <typename TokenT, typename IndexT>
void FullScheduler<TokenT, IndexT>::process_inflight_results(const Output& out,
                                                          std::vector<int>& finished_idx) {
    // 与 process_results 的差别：
    //   - commit 已在 submit 时做完（step_overlap 阶段 3），这里只做 record_predicted。
    //   - prefill reqs 已在 submit 时搬到 running_（step_overlap 阶段 4），
    //     这里不再 splice。
    //   - 多一条 abort 分支：inflight 期间被 abort_req 标 Finished 的 req，
    //     跳过 record_predicted 直接进 finished_idx 让 free_finished 清理。
    vt_assert((int)out.next_tokens.size() == cur_batch_.size(),
              "FullScheduler::process_inflight_results: next_tokens size must == batch size");

    const Token eos     = engine_->eos_token_id();
    const int   max_out = config_.default_max_output;

    for (int i = 0; i < cur_batch_.size(); ++i) {
        Req&   r   = *cur_batch_.reqs[i];
        Token  tok = out.next_tokens[i];

        // inflight 期间被 abort：state 已切到 Finished。GPU 写过的 KV 仍然有效
        // （不能浪费——free_finished 会把它写回 radix 给后续 req 复用）。
        if (r.state == Req::Finished) {
            finished_idx.push_back(i);
            continue;
        }

        // record_predicted 自带状态切换：Prefill → Decode（首次 decode 后稳定为 Decode）。
        r.record_predicted(tok);

        bool hit_eos = (eos != Token(-1)) && (tok == eos);
        bool hit_max = (int)r.output.size() >= max_out;

        if (hit_eos || hit_max) {
            r.finish();
            finished_idx.push_back(i);
        }
    }
}

// ============================================================================
// 内部：归还 finished 资源
// ============================================================================

template <typename TokenT, typename IndexT>
void FullScheduler<TokenT, IndexT>::free_finished(const std::vector<int>& finished_idx) {
    if (finished_idx.empty()) return;

    // finished req 此时已在 running_ 里：prefill 完成的 req 在 process_results
    // 末尾 splice 过去了；decode 完成的本来就在 running_
    for (int idx : finished_idx) {
        Req& r = *cur_batch_.reqs[idx];

        // 拼 tokens：input[0..prefill_pos) + output[0..decode_pos)
        // 注意 prefill_pos 已经被 process_results 推到 input.size()，
        // decode_pos 已经 +1（包含本步采的 token）。
        std::vector<Token> tokens;
        tokens.reserve((size_t)r.cached_len());
        tokens.insert(tokens.end(), r.input.begin(),
                                    r.input.begin() + r.prefill_pos);
        tokens.insert(tokens.end(), r.output.begin(),
                                    r.output.begin() + r.decode_pos);

        typename Cache::FinishInput fi{
            /*table_idx*/      r.table_idx,
            /*prepare_result*/ {r.node, r.radix_hit_len},
            /*cur_cached_len*/ r.cached_len(),
            /*tokens*/         std::move(tokens),
        };
        cache_->finished(fi);

        auto it = find_in_running(r.id);
        vt_assert(it != running_.end(),
                  "FullScheduler::free_finished: finished req missing in running_");
        running_.erase(it);
    }
}

// ============================================================================
// explicit instantiation — default 类型，对齐 Request<> / CacheManager<> 等
// ============================================================================
template class FullScheduler<int32_t, int32_t>;

} // namespace vt
