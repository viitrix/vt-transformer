// schedule.cpp — see schedule.hpp
//
// Compile (from core/):
//   g++ -std=c++17 -O2 -Wall -I.. -I../opt schedule.cpp -c -o /tmp/schedule.o
// Link against libzmq (-lzmq).

#include "schedule.hpp"

#include <algorithm>
#include <iostream>
#include <utility>
#include <variant>

namespace minisgl {

Scheduler::Scheduler(SchedulerConfig cfg, std::unique_ptr<IForwardBackend> backend)
    : cfg_(std::move(cfg)),
      mq_(cfg_.bind_pull_addr, cfg_.connect_push_addr),
      cm_(cfg_.cache_page_size, cfg_.cache_num_pages),
      tm_(cfg_.max_running_reqs),
      decode_manager_(cfg_.cache_page_size),
      prefill_manager_(cm_, tm_, decode_manager_, live_reqs_, cfg_.max_seq_len),
      backend_(std::move(backend)),
      sched_stream_(backend_->sched_stream_event()),
      engine_stream_(backend_->engine_stream_event()) {
    std::cerr << "[scheduler] PULL bind " << cfg_.bind_pull_addr
              << ", PUSH connect " << cfg_.connect_push_addr
              << ", page_size=" << cfg_.cache_page_size
              << " num_pages=" << cfg_.cache_num_pages
              << " max_running=" << cfg_.max_running_reqs
              << " max_extend_tokens=" << cfg_.max_extend_tokens
              << " max_seq_len=" << cfg_.max_seq_len
              << " overlap=" << (cfg_.disable_overlap_scheduling ? "off" : "on")
              << "\n";
}

// ============================================================================
// run_forever
// ============================================================================

void Scheduler::run_forever() {
    std::cerr << "[scheduler] main loop start\n";
    try {
        if (cfg_.disable_overlap_scheduling) {
            for (;;) normal_loop();
        } else {
            // overlap_loop:上一轮的 ForwardData 延后一轮处理。data 在迭代间传递。
            ForwardData* data = nullptr;
            for (;;) data = overlap_loop(data);
        }
    } catch (const SchedulerExit&) {
        std::cerr << "[scheduler] ExitMsg received, main loop exit\n";
    }
}

// ============================================================================
// overlap_loop / normal_loop
// ============================================================================

ForwardData* Scheduler::overlap_loop(ForwardData* last) {
    bool blocking = !(last != nullptr
                      || prefill_manager_.runnable()
                      || decode_manager_.runnable());
    drain_msgs(blocking);

    ForwardInput* input = schedule_next_batch();
    ForwardData* ongoing = nullptr;
    if (input) {
        // 让 engine_stream 等 sched_stream 最近的 record(本版 scheduler 全 CPU
        // 工作,sched_stream 上没真 CUDA 工作,event 立即 fire;接 scheduler-side
        // CUDA 工作时此处变实质性同步)。
        sched_stream_->record();
        engine_stream_->wait_on(sched_stream_);
        ForwardOutput out = backend_->forward(*input);
        ongoing = new ForwardData{std::move(*input), std::move(out)};
        delete input;
    }
    // 处理上一轮(可能在本轮 forward 启动后才完成 GPU 工作)。
    process_last_data(last);
    delete last;
    return ongoing;
}

void Scheduler::normal_loop() {
    bool blocking = !(prefill_manager_.runnable() || decode_manager_.runnable());
    drain_msgs(blocking);

    ForwardInput* input = schedule_next_batch();
    if (!input) return;
    // normal_loop 当轮 forward 当轮处理,不延后。record/wait_on 仍调以保持代码
    // 路径一致(sched_stream 没 CUDA 工作,实际 no-op)。
    sched_stream_->record();
    engine_stream_->wait_on(sched_stream_);
    ForwardOutput out = backend_->forward(*input);
    ForwardData data{std::move(*input), std::move(out)};
    delete input;
    process_last_data(&data);
}

// ============================================================================
// schedule_next_batch + prepare_batch + allocate_paged
// ============================================================================

ForwardInput* Scheduler::schedule_next_batch() {
    // continue batch 策略:prefill 先,decode 后(对应 python scheduler.py:219-225)。
    std::unique_ptr<Batch> pb = prefill_manager_.schedule_next_batch(cfg_.max_extend_tokens);
    if (!pb) pb = decode_manager_.schedule_next_batch();
    if (!pb) return nullptr;
    allocate_paged(*pb);
    ForwardInput fi = prepare_batch(*pb);
    return new ForwardInput(std::move(fi));
}

ForwardInput Scheduler::prepare_batch(Batch& b) {
    ForwardInput fi;
    fi.batch.phase = b.phase;
    fi.batch.reqs  = b.reqs;  // shared_ptr copy

    size_t total_extend = 0;
    for (const auto& r : b.reqs) total_extend += r->extend_len();

    fi.batch.input_ids.clear();
    fi.batch.input_ids.reserve(total_extend);
    fi.batch.positions.clear();
    fi.batch.positions.reserve(total_extend);
    fi.input_mapping.clear();
    fi.input_mapping.reserve(total_extend);
    fi.write_mapping.clear();
    fi.write_mapping.reserve(b.reqs.size());
    fi.write_lens.clear();
    fi.write_lens.reserve(b.reqs.size());

    fi.sample_args.temperatures.clear();
    fi.sample_args.top_k.clear();
    fi.sample_args.top_p.clear();
    fi.sample_args.ignore_eos.clear();

    for (const auto& r : b.reqs) {
        const size_t extend = r->extend_len();
        // batch.input_ids: r.input_ids 的 [cached_len, device_len) 段
        for (size_t i = r->cached_len; i < r->device_len; ++i) {
            fi.batch.input_ids.push_back(r->input_ids[i]);
        }
        // positions: arange(cached_len, device_len)
        for (size_t i = r->cached_len; i < r->device_len; ++i) {
            fi.batch.positions.push_back(static_cast<int64_t>(i));
        }
        // input_mapping: extend_len 个 r->table_idx
        for (size_t i = 0; i < extend; ++i) {
            fi.input_mapping.push_back(r->table_idx);
        }
        // write_mapping / write_lens: 每 req 一项
        fi.write_mapping.push_back(r->table_idx);
        fi.write_lens.push_back(r->can_decode() ? static_cast<int64_t>(r->device_len) : -1);

        // sample_args
        fi.sample_args.temperatures.push_back(r->sampling_params.temperature);
        fi.sample_args.top_k.push_back(r->sampling_params.top_k);
        fi.sample_args.top_p.push_back(r->sampling_params.top_p);
        fi.sample_args.ignore_eos.push_back(r->sampling_params.ignore_eos);
    }
    return fi;
}

void Scheduler::allocate_paged(Batch& b) {
    const size_t page_size = cm_.page_size();
    for (const auto& req : b.reqs) {
        size_t first_page = div_ceil(req->cached_len, page_size);
        size_t last_page  = div_ceil(req->device_len, page_size);
        if (last_page <= first_page) continue;
        size_t need = last_page - first_page;
        std::vector<int32_t> alloc = cm_.allocate(need);
        // 把 per-token offsets 展开写入 req->page_indices
        for (size_t i = 0; i < need; ++i) {
            int32_t base = alloc[i];
            for (size_t j = 0; j < page_size; ++j) {
                size_t pos = (first_page + i) * page_size + j;
                if (pos < req->page_indices.size()) {
                    req->page_indices[pos] = base + static_cast<int32_t>(j);
                }
            }
        }
    }
}

// ============================================================================
// process_last_data
// ============================================================================

void Scheduler::process_last_data(ForwardData* d) {
    if (!d) return;
    d->second.copy_done->synchronize();

    std::vector<DetokenizeMsg> reply;
    std::vector<std::shared_ptr<Req>> just_prefilled;
    std::set<int64_t> new_finished_uids;

    Batch& batch = d->first.batch;
    const bool is_prefill = batch.is_prefill();

    for (size_t i = 0; i < batch.reqs.size(); ++i) {
        auto& req = batch.reqs[i];
        int32_t tok = d->second.next_tokens_cpu[i];

        // append_host + complete_one(对应 python req.append_host + 隐式 complete_one
        // 在 engine.forward_batch 里)。
        req->append_host(tok);
        req->complete_one();

        bool finished = !req->can_decode()
                     || (!req->sampling_params.ignore_eos && tok == cfg_.eos_token_id);

        reply.push_back(DetokenizeMsg{req->uid, tok, finished});

        if (finished) {
            // overlap 调度可能让一个 req 同时在 ongoing_N 和 batch_{N+1} 中,
            // 它的 finished 会被处理两次;第二次跳过 free(对应 python
            // scheduler.py:159 的 skip second free)。
            if (finished_uids_.count(req->uid) == 0) {
                decode_manager_.remove_req(req);
                free_req_resources(req);
                new_finished_uids.insert(req->uid);
            }
        } else if (is_prefill) {
            cache_req(*req, /*finished=*/false);
            just_prefilled.push_back(req);
        }
        // decode 非 finished:python 在 _process_last_data 里也不动它。
    }

    decode_manager_.filter_reqs(just_prefilled);
    send_result(reply);

    finished_uids_ = std::move(new_finished_uids);
}

// ============================================================================
// cache_req / free_req_resources
// ============================================================================

void Scheduler::cache_req(Req& req, bool finished) {
    // insert_ids / page_indices 都按 req.cached_len 截断。
    std::vector<int32_t> insert_ids(req.input_ids.begin(),
                                     req.input_ids.begin() + req.cached_len);
    std::vector<int32_t> page_indices_for_insert(req.page_indices.begin(),
                                                  req.page_indices.begin() + req.cached_len);

    ReqCacheHandle old_handle = req.cache_handle;
    auto ins = cm_.insert_prefix(insert_ids, page_indices_for_insert);
    size_t new_cached = ins.cached_len;
    ReqCacheHandle new_handle = ins.handle;

    // 解掉旧 handle 的锁。
    cm_.lock_handle(old_handle, /*unlock=*/true);

    // [old_handle.cached_len, new_cached) 是这一轮才发现"早被别的请求 cache 了"
    // 的部分;对应 python cache.py:73 的 self._free。
    if (new_cached > old_handle.cached_len) {
        std::vector<int32_t> already_cached(
            page_indices_for_insert.begin() + old_handle.cached_len,
            page_indices_for_insert.begin() + new_cached);
        cm_.free(already_cached);
    }

    if (finished) {
        // [new_handle.cached_len, req.cached_len) 是不能页对齐而被砍掉的尾巴;
        // finished 时直接 free(python cache.py:75-76)。
        if (req.cached_len > new_handle.cached_len) {
            std::vector<int32_t> tail(
                page_indices_for_insert.begin() + new_handle.cached_len,
                page_indices_for_insert.end());
            cm_.free(tail);
        }
    } else {
        // 切到新 handle 并锁住。
        req.cache_handle = new_handle;
        cm_.lock_handle(new_handle);
    }
}

void Scheduler::free_req_resources(const std::shared_ptr<Req>& req) {
    if (req->table_idx >= 0) tm_.free(req->table_idx);
    cache_req(*req, /*finished=*/true);
    live_reqs_.erase(req->uid);
    // 此处 shared_ptr 引用计数 -1。如果 batch 或 decode_manager 还持着,
    // Req 不会被析构;直到它们都释放才回收。
}

// ============================================================================
// 消息处理
// ============================================================================

void Scheduler::drain_msgs(bool blocking) {
    if (blocking) {
        process_one_msg(mq_.recv());
    }
    while (auto m = mq_.try_recv()) {
        process_one_msg(*m);
    }
}

void Scheduler::send_result(const std::vector<DetokenizeMsg>& reply) {
    for (const auto& m : reply) mq_.send(m);
}

void Scheduler::process_one_msg(const BackendMsg& m) {
    std::visit(overloaded{
        [this](const UserMsg& um) {
            // max_seq_len clamp/drop 由 PrefillManager.add_one_req 内部处理。
            prefill_manager_.add_one_req(um);
        },
        [this](const AbortBackendMsg& am) {
            auto r = prefill_manager_.abort_req(am.uid);
            if (!r) r = decode_manager_.abort_req(am.uid);
            if (r) free_req_resources(r);
        },
        [this](const ExitMsg&) {
            throw SchedulerExit{"ExitMsg received"};
        },
        [this](const BatchBackendMsg& bm) {
            for (const auto& item : bm.data) process_one_item(item);
        },
    }, m);
}

void Scheduler::process_one_item(const BackendMsgItem& item) {
    std::visit(overloaded{
        [this](const UserMsg& um) { prefill_manager_.add_one_req(um); },
        [this](const AbortBackendMsg& am) {
            auto r = prefill_manager_.abort_req(am.uid);
            if (!r) r = decode_manager_.abort_req(am.uid);
            if (r) free_req_resources(r);
        },
        [this](const ExitMsg&) { throw SchedulerExit{"ExitMsg received"}; },
    }, item);
}

}  // namespace minisgl
