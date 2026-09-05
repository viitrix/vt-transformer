// qwen3.hpp — Qwen3Engine：加载权重指针表的引擎外壳。
//
// 生命周期：main 构造后调用 init_comm() 跑 dag/init.vt，在 env_->hash() 里建
// tensor 表并抽出 device ptr 填进 comm_。ZMQ 通道不在本类——由 backend_cuda.cpp
// 的 init_ipc() 负责。prefill.h::prefill_forward 等桥接层经由 env() / comm() 取 env 与权重表。

#ifndef _QWEN3_QWEN3_HPP_
#define _QWEN3_QWEN3_HPP_

#include <utility>

#include <core/vt.hpp>

#include "common.cuh"

namespace qwen3 {

struct Request {
    int rid = 0;
    bool is_abort = false;
    std::vector<int32_t> input_ids;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int max_new_tokens = 128;
    std::vector<int32_t> stop_token_ids;
    bool stream = true;
};

// 与前端 output 消息的载荷一一对应（"type":"output" 由 IPC 层补上）：
//   {"rid":N,"tokens":[..],"finish":null|"stop"|"length"}
struct Response {
    int rid = 0;

    // 本条消息新增的 token ids（增量，非全量）；可为空（如仅发结束标记）
    std::vector<int32_t> tokens;

    // 空 = 生成中（序列化为 null）；"stop" = 命中停止条件；"length" = 达到 max_new_tokens
    std::string finish;
};

// 槽位生命周期：IDLE → PREFILL（已受理待算 prefill）→ DECODE（逐 token 生成）
// → STOP（命中停止条件 / 达到 max_new_tokens，待发最终响应）或 ABORT（客户端
// 撤销，待清理）；STOP / ABORT 收尾完成后回到 IDLE，行内 KV 残留仍供前缀复用。
enum KVSlotStatus {
    IDLE = 0,
    PREFILL = 1,
    DECODE = 2,
    STOP = 3,
    ABORT = 4,
};

// 槽位在途登记：除状态外记录 rid，abort 消息靠 rid 定位要撤销的行。
struct KVSlot {
    KVSlotStatus status = IDLE;
    int rid = -1;  // 占用该行的请求 id；IDLE 行无归属
};

class Qwen3Engine {
public:
    explicit Qwen3Engine(vt::Enviroment* env);

    // env_ 非拥有，comm_ 全是不拥有的 device 指针。
    ~Qwen3Engine();

    void init();
    void forward(std::vector<Request>& incoming);
    std::optional<Response> process_last();

    vt::Enviroment& env()  { return *env_; }
    CommonArgs&     comm() { return comm_; }

private:
    void init_comm();

    // 在 IDLE 槽位里挑缓存前缀与 req.input_ids 公共前缀最长的一行（复用已落 KV），
    // 返回 {槽位行号, 前缀长度}；槽位全 BUSY 时行号为 -1。
    std::pair<int, int> find_best_slot(const Request& req) const;

    bool run_dag(const char* fileName);

private:
    vt::Enviroment* env_ = nullptr;  // 非拥有：CUDA device / stream / DAG 都从它取
    CommonArgs comm_;

    // KV Cache 槽位表：下标即 token_table 的行号（table_idx），
    // forward 分配行时置 PREFILL 并登记 rid，行内进度看 cached_lens_cpu。
    KVSlot slots_[kMaxRunningReqs];

};

} // namespace qwen3

#endif // _QWEN3_QWEN3_HPP_
