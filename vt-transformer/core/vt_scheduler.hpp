#ifndef _VT_SCHEDULER_HPP_
#define _VT_SCHEDULER_HPP_

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <vector>

#include "vt.hpp"
#include "vt_cache.hpp"
#include "vt_request.hpp"

namespace vt {

// FullScheduler 配置（对齐 mini-sglang SchedulerConfig 中"计算相关"的字段；
// 网络层 zmq 地址等不在此处——FullScheduler 与 Endpoint 解耦的边界）。
//
// 模板参数：把 Token / Index 作为类型参数放在这里——FullScheduler 跟着用同样的
// 两个参数实例化（见下方 FullScheduler 模板）。default 是 int32_t / int32_t，
// 对齐 Request<> / CacheManager<> 等其它模板族的默认。
//
// 用法：
//   SchedulerConfig<>                      cfg;           // 默认 int32_t/int32_t
//   SchedulerConfig<int16_t, int32_t>      cfg16;         // 自定义
//   FullScheduler<int16_t, int32_t>            sched(cfg16, &engine);
template <typename TokenT = int32_t, typename IndexT = int32_t>
struct SchedulerConfig {
    using Token = TokenT;
    using Index = IndexT;

    int max_running_reqs   = 128;    // PageTable 行数上限 = 同时活跃 req 数
    int num_pages          = 1024;   // KV pool page 总数
    int max_seq_len        = 4096;   // 单条 req 的 token 上限（input + output）
    int page_size          = 1;      // 每个 page 含多少 token
    int max_extend_tokens  = 8192;   // 单次 prefill batch 的 token budget（chunked 上限）
    int default_max_output = 1024;   // 单条 req 默认 output 上限（sampling_params 未接入前的兜底）
};

// 一次 forward 提交给 Engine 的内容（对齐 mini-sglang core.py:71-94）。
//
// reqs：非拥有指针，所有权属于 FullScheduler 的 pending_ / running_。
// phase：Prefill / Decode 二选一——本实现沿用 mini-sglang 的"prefill OR decode"
//        单 batch 策略（scheduler.py:219-225），不在同一 batch 内混跑。
template <typename Token = int32_t, typename Index = int32_t>
struct Batch {
    using Req = Request<Token, Index>;

    enum Phase { Prefill, Decode };

    std::vector<Req*> reqs;
    Phase             phase = Prefill;

    bool is_prefill() const { return phase == Prefill; }
    bool is_decode()  const { return phase == Decode;  }
    int  size()       const { return (int)reqs.size(); }
};

// 一次 forward 的产出（对齐 mini-sglang engine.py:23-27 的 ForwardOutput）。
// next_tokens 长度必须 == batch.size，与 batch.reqs 同序。
template <typename Token = int32_t>
struct ForwardOutput {
    std::vector<Token> next_tokens;
};

// forward_async 的回执。默认 sync 路径直接把 Output 装在 handle 里；
// 真实 async engine 子类化 EngineBase 时可以重写 forward_async/wait，
// 用自己的 CUDA event / stream 状态替换 sync_output 字段（直接忽略它）。
template <typename Token = int32_t>
struct ForwardHandle {
    bool                valid = false;
    ForwardOutput<Token> sync_output;   // sync 路径专用
};

// Engine 抽象基类——FullScheduler 把 batch 喂进来，子类返回新 token。
// 真实实现（CudaEngine / MockEngine 等）子类化后注入 FullScheduler 构造函数。
// 对齐 mini-sglang engine.py:29 的 Engine.forward_batch。
template <typename Token = int32_t, typename Index = int32_t>
class EngineBase {
public:
    using BatchT         = Batch<Token, Index>;
    using ForwardOutputT = ForwardOutput<Token>;
    using ForwardHandleT = ForwardHandle<Token>;

    virtual ~EngineBase() = default;

    // 同步 forward：读取 batch 里每个 req 的 [cached_len, total_len) 段作为输入，
    // 把新 token 写到 KV（位置由 PageTable 决定），返回每个 req 一个新 token。
    // sync 路径（FullScheduler::step）走这条。
    virtual ForwardOutputT forward(const BatchT& batch) = 0;

    // 异步 forward：立即返回 handle，不阻塞；调用方在需要结果时再调 wait。
    // 默认实现就是包一层同步调用——sync engine 不需要重写。
    // 真实 async engine 重写 forward_async（提交到自己的 CUDA stream 立刻返回）
    // 和 wait（在调用方需要结果时同步 stream / event）。
    virtual ForwardHandleT forward_async(const BatchT& batch) {
        return ForwardHandleT{/*valid=*/true, forward(batch)};
    }
    virtual ForwardOutputT wait(ForwardHandleT&& h) {
        vt_assert(h.valid, "EngineBase::wait: handle not valid");
        return std::move(h.sync_output);
    }

    // EOS token id；decode 出该 id 视为完成。默认 -1 表示"不靠 EOS 终止"，
    // 仅靠 max_output_len 触发 Finished。
    virtual Token eos_token_id() const { return Token(-1); }
};

// FullScheduler：推理计算的核心调度器。
//
// Continue Batch（连续 batching）模型（对齐 mini-sglang scheduler.py:219-225）：
//   - 每个 step 选 prefill batch 或 decode batch 之一（prefill 优先）
//   - Decode req 跨 step 保留在 running_ 中（不被新 prefill 抢占赶出）
//   - Finished req 在 step 末尾从 running_ 移除并归还资源
//
// 职责切分：
//   - FullScheduler  ：req 状态机驱动、batch 调度、CacheManager 协调
//   - CacheManager：radix 命中、page 分配 / 回收（已实现）
//   - EngineBase ：实际 model forward（子类提供）
//   - Endpoint   ：与 frontend 的 ZMQ IO（已实现，由外部 driver 驱动）
//
// 模板参数与 SchedulerConfig 对齐：TokenT / IndexT 默认 int32_t / int32_t。
// 实例化 .cpp 末尾对 default 类型做了 explicit instantiation，default 类型
// 直接用 `FullScheduler<>` 即可；其它类型需要在自己的代码里再 instantiate。
//
// 非线程安全：单线程驱动，与 mini-sglang scheduler 单线程模型一致。
template <typename TokenT = int32_t, typename IndexT = int32_t>
class FullScheduler {
public:
    using Token  = TokenT;
    using Index  = IndexT;
    using Req    = Request<Token, Index>;
    using BatchT = Batch<Token, Index>;
    using Cache  = CacheManager<Token, Index>;
    using Engine = EngineBase<Token, Index>;
    using Output = ForwardOutput<Token>;
    using Handle = ForwardHandle<Token>;
    using Config = SchedulerConfig<Token, Index>;

    // FullScheduler 拥有 CacheManager；Engine 由外部注入（不拥有）。
    FullScheduler(Config config, Engine* engine);

    // ---- 外部 IO 入口（Endpoint / 测试驱动）----

    // 入队新 req（state=Waiting）。table_idx / radix 节点等资源在入选 prefill 时分配。
    // 传入 Req 按值移动；id / input 由调用方填好（output 留空）。
    void add_req(Req req);

    // 取消未完成的 req；若已分配资源（已入选 prefill）则归还。找不到返回 false。
    bool abort_req(uint64_t uid);

    // ---- 调度循环 ----

    // 跑一步：选 batch → 准备资源 → forward → 处理结果 → 归还 finished 资源。
    // 无 work（pending / running 都空）时返回 ran_batch=false。
    //
    // results：本步 forward 给 batch 里每个 req 产出的"新 token + 是否完成"。
    // 调用方（典型是 backend driver）按这把它翻译成 Endpoint::Result 回送 frontend。
    // sync step() 中 results 跟 ran_batch 描述同一次 forward；step_overlap() 中
    // results 描述 phase 1 处理的上一次 inflight，而 ran_batch 描述 phase 2 本次提交——
    // 与 finished_count 的语义一致（详见 step_overlap 注释）。
    struct ReqResult {
        uint64_t uid;         // 对应 Request::id
        Token    next_token;  // 本步 engine 预测出的 token
        bool     finished;    // 本步是否触发 finish（EOS / max_output / inflight 期间 abort）
    };
    struct StepResult {
        bool                     ran_batch;      // 本步是否真正跑了一次 forward
        int                      finished_count; // 本步 finished 的 req 数
        std::vector<ReqResult>   results;        // empty 当 ran_batch==false（sync）/ phase 1 没有 inflight（overlap）
    };
    StepResult step();

    // Overlap 版本：把"处理上一步结果"和"提交本轮 forward"重叠。
    //
    // 单次调用做两件事：
    //   阶段 1（如果上次有 in-flight）：wait 上次 forward → record token → free finished
    //   阶段 2：选 batch → prepare → speculative commit → forward_async（不阻塞）
    //
    // 阶段 1 处理的是"上一次"的结果，所以 finished_count 反映的是上一步的 forward 产出，
    // 而 ran_batch 反映的是"本次是否提交了新 forward"。两者描述的不是同一个 batch。
    //
    // 冷启动：第一次调用没有 in-flight，跳过阶段 1。
    // 收尾：has_work() == false 但 has_inflight() == true 时，再调一次 drain 最后一批。
    //
    // 不要和 sync 的 step() 混用——commit/record 时序不同会导致 req 状态错乱。
    StepResult step_overlap();

    bool has_work() const { return !pending_.empty() || !running_.empty(); }

    // 是否有 in-flight forward（仅 overlap 模式有意义）；用于判断是否需要 drain。
    bool has_inflight() const { return !inflight_batch_.reqs.empty(); }

    // ---- 容量查询（诊断用）----
    int    pending_count() const { return (int)pending_.size(); }
    int    running_count() const { return (int)running_.size(); }
    Cache& cache()                { return *cache_; }

private:
    // 用 std::list 是为了在 splice / erase 时 Req* 保持有效（Batch 持有 Req*）。
    using ReqList = std::list<Req>;
    using ReqIter = typename ReqList::iterator;

    Config                     config_;
    Engine*                    engine_;  // 不拥有
    std::unique_ptr<Cache>     cache_;

    ReqList                    pending_; // state=Waiting，等待 prefill
    ReqList                    running_; // state=Prefill-完成 / Decode，跨 step 保留

    // 一次 step 内构造的 batch——成员而非局部变量，避免每 step 重分配。
    // step() 末尾 clear；reqs 仅持指针，无所有权。
    BatchT                     cur_batch_;

    // overlap 模式专用：上次 step_overlap 提交、还在 engine 里跑的 batch。
    // 下次 step_overlap 进入阶段 1 时被处理（wait + record + free_finished）。
    BatchT                     inflight_batch_;
    Handle                     inflight_handle_;

    // pending_ / running_ 中找 uid；找到返回迭代器，否则 end()。
    ReqIter find_in_pending(uint64_t uid);
    ReqIter find_in_running(uint64_t uid);

    // 把 pending_ 里能在 token budget 内装下的 req 选进 batch。
    // 返回选中的 req 数；0 表示没装下任何 req。
    int  try_schedule_prefill();

    // 把整个 running_ 拼成 decode batch。running_ 空时返回 false。
    bool try_schedule_decode();

    // 提交 batch 前：prefill 走 alloc_row + CacheManager.prepare；
    //               decode 不重 prepare（只 allocate_pages）。
    // 失败时已动的资源全部回滚（prefill 路径），返回 false。
    bool prepare_batch();

    // forward 完成后：把新 token append 到 req.output；判定 finished；
    // prefill 完成的 req 从 pending_ 转入 running_（设 state=Decode）。
    // finished_idx 收集本步 finished 的 req 在 batch.reqs 里的下标。
    // sync 路径用——做 commit + record 两步。
    void process_results(const Output& out, std::vector<int>& finished_idx);

    // overlap 路径用——commit 已在 submit 时做完，这里只做 record + 状态切换。
    // 其它流程（判定 finished、splice pending→running）跟 process_results 一致。
    void process_inflight_results(const Output& out, std::vector<int>& finished_idx);

    // finished_idx 对应的 req：CacheManager.finished + 从 running_ 移除。
    void free_finished(const std::vector<int>& finished_idx);

    // 把一次 prepare 的副作用反向：unlock radix 路径 + free row + 把 Req 重置到 Waiting。
    // 用于 prepare_batch 失败时回滚已 prepare 的 req，让它留在 pending_ 等下次重试。
    void rollback_prepare_(Req* r);

    // overlap 模式下：req 是否还在 inflight_batch_ 里（已 submit、未 process）。
    // abort_req 用它判定能不能立刻 erase——inflight 期间不能动，否则 inflight_batch_
    // 里的指针会悬挂。
    bool is_inflight_req_(const Req* r) const;
};

} // namespace vt

#endif
