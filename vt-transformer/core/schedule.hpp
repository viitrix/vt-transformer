// schedule.hpp — Scheduler(完整版,带 continue batch + overlap loop)
//
// 端口 mini-sglang/python/minisgl/scheduler/scheduler.py:Scheduler。
// 主要结构:
//   * run_forever        按 cfg_.disable_overlap_scheduling 选 overlap_loop / normal_loop
//   * overlap_loop       上一轮的 ForwardData 延后一轮处理,与当前 forward 在
//                        engine_stream_ 上并行
//   * normal_loop        当轮 forward 当轮处理,不延后
//   * schedule_next_batch  prefill_manager 先,decode_manager 后(prefill-first)
//   * process_last_data  消费上一轮 output:append_host + complete_one + cache_req
//   * process_one_msg    UserMsg -> prefill_manager.add_one_req
//                        AbortBackendMsg -> prefill/decode abort + free
//                        ExitMsg -> throw SchedulerExit
//                        BatchBackendMsg -> 递归
//
// 组合模式:Scheduler 持 std::unique_ptr<IForwardBackend>,由用户构造注入。
// 真实 GPU backend 继承 IForwardBackend,在 forward() 实现里走模型。
//
// 所有用到 Req 的地方都走 shared_ptr(见 req.hpp 文件头注释):overlap 模式下
// 同一个 Req 可能同时存在于 ongoing_N 与 batch_{N+1} 中,shared_ptr 让它的
// 生命周期跟随引用计数自然结束。
//
// overlap 同步依赖真实的 CUDA stream/event:Sched/Engine 两条 stream 通过
// SyncEvent 协调。必须 MINISGL_USE_CUDA,否则编译失败;Scheduler 直接复用
// backend 提供的两个 SyncEvent*。

#ifndef MINISGL_SCHEDULE_HPP
#define MINISGL_SCHEDULE_HPP

#ifndef MINISGL_USE_CUDA
#error "schedule.hpp requires MINISGL_USE_CUDA"
#endif

#include <cstddef>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "cache.hpp"
#include "config.hpp"
#include "decode.hpp"
#include "forward.hpp"
#include "message.hpp"
#include "prefill.hpp"
#include "sync.hpp"
#include "table.hpp"

namespace minisgl {

// ExitMsg 抛出,run_forever 顶层 catch。
struct SchedulerExit : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Scheduler {
public:
    Scheduler(SchedulerConfig cfg, std::unique_ptr<IForwardBackend> backend);

    Scheduler(const Scheduler&)            = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // 进入主循环,直到收到 ExitMsg 才返回。
    void run_forever();

private:
    using CacheMgr = CacheManager<int32_t, int32_t, HostAllocator>;

    // ---- 主循环 ----
    ForwardData* overlap_loop(ForwardData* last);
    void         normal_loop();

    // ---- 调度 ----
    // continue batch 入口:prefill_manager 先,decode_manager 后。
    // 没有可跑的 batch 时返回 nullptr。
    ForwardInput* schedule_next_batch();

    // 端口 scheduler.py:_prepare_batch。填充 batch.input_ids/positions,
    // 构造 input_mapping/write_mapping/write_lens/sample_args。
    ForwardInput prepare_batch(Batch& b);

    // 端口 cache.py:allocate_paged。给 batch 里每个 req 在 [cached_len, device_len)
    // 区间补 page,把分配到的 page id 展开写入 req.page_indices。
    void allocate_paged(Batch& b);

    // ---- 结果处理 ----
    // 端口 scheduler.py:_process_last_data。同步 copy_done,把每 req 的 next_token
    // 追加进 input_ids + complete_one,判定 finished,触发 cache_req 与 free。
    // finished_uids_ 用来跳过 overlap 重复 free(对应 python self.finished_reqs)。
    void process_last_data(ForwardData* d);

    // ---- 消息 ----
    void process_one_msg(const BackendMsg& m);
    void process_one_item(const BackendMsgItem& item);
    void drain_msgs(bool blocking);
    void send_result(const std::vector<DetokenizeMsg>& reply);

    // ---- 资源管理 ----
    // 端口 cache.py:cache_req。把 req.input_ids[:cached_len] + 对应 page_indices
    // 插进 RadixCache,处理 dedup free / tail free / handle 切换。
    void cache_req(Req& req, bool finished);

    // 释放 req 占的 table slot + cache 资源,从 live_reqs_ 移除。
    // shared_ptr 让 Req 在 batch 引用消失前不被析构。
    void free_req_resources(const std::shared_ptr<Req>& req);

    // ---- 工具 ----
    static size_t div_ceil(size_t a, size_t b) noexcept {
        return (a + b - 1) / b;
    }

    // ---- 成员 ----
    // 注意声明顺序 = 构造顺序:cfg_ -> mq_ -> cm_ -> tm_ -> live_reqs_ ->
    // decode_manager_ -> prefill_manager_(持 decode_manager_ 与 live_reqs_ 引用)
    // -> backend_ -> sched/engine_stream_(从 backend 取 non-owning 指针)
    // -> finished_uids_。
    SchedulerConfig                                  cfg_;
    BackendMQ                                        mq_;
    CacheMgr                                         cm_;
    TableManager                                     tm_;
    std::unordered_map<int64_t, std::shared_ptr<Req>> live_reqs_;
    DecodeManager                                    decode_manager_;
    PrefillManager                                   prefill_manager_;
    std::unique_ptr<IForwardBackend>                 backend_;
    // MINISGL_USE_CUDA 下由 backend 暴露的真实 stream event;指针归 backend
    // 所有,Scheduler 只持 non-owning 引用。backend 必须保证两者非空。
    SyncEvent*                                       sched_stream_;
    SyncEvent*                                       engine_stream_;
    // 上一轮 process_last_data finished 的 uid 集合,本轮跳过它们的 free。
    // 对应 python scheduler.py:68 的 self.finished_reqs。
    std::set<int64_t>                                finished_uids_;
};

}  // namespace minisgl

#endif  // MINISGL_SCHEDULE_HPP
