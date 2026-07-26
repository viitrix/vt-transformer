#ifndef _VT_REQUEST_HPP_
#define _VT_REQUEST_HPP_

#include <cstdint>
#include <vector>

#include "vt.hpp"
#include "vt_radix.hpp"

namespace vt {

// 一次推理请求的生命周期：
//   Waiting  -> 已入队，尚未被调度
//   Prefill  -> prompt 正在写入 KV（可能分块）
//   Decode   -> 逐 token 生成，output 每步 +1
//   Finished -> 命中 EOS 或达到上限，无更多工作
//
// 位置语义（与 mini-sglang 的 cached_len / device_len 对齐）：
//   prefill_pos : input 中已落 KV 的 token 数
//   decode_pos  : output 中已落 KV 的 token 数（注意：output 末尾那个 token
//                 是 forward 刚预测出来的，KV 还没写——它的 KV 由下一次 forward 写。
//                 所以 output.size() == decode_pos + 1 永远成立， Decode 状态下）
//   cached_len  = prefill_pos + decode_pos   ← 这些 slot 的 KV 已写、可复用
//   total_len   = input.size() + output.size()
//   extend_len  = total_len - cached_len   （本次 forward 待写入的量）
//
// 例（无 prefix 命中）：
//   prefill forward 写 [0, input.size()) 的 KV，预测 T1
//     → prefill_pos = input.size(), decode_pos = 0, output = [T1]
//   第一次 decode forward 写 T1 的 KV 到 slot input.size()，预测 T2
//     → prefill_pos 不变, decode_pos = 1, output = [T1, T2]
//   第二次 decode forward 写 T2 的 KV 到 slot input.size()+1，预测 T3
//     → decode_pos = 2, output = [T1, T2, T3]
//
// 模板参数（与 PageTable / CacheManager / RadixNode 对齐）：
//   TokenT  token id 类型，存进 input / output
//   IndexT  page_id / table_idx 等下标类型
//
// 状态迁移由 scheduler 显式驱动；Request 自身只负责薄记。
template <typename TokenT = int32_t, typename IndexT = int32_t>
struct Request {
    using Token = TokenT;
    using Index = IndexT;
    using Node  = RadixNode<Token, Index>;

    enum State {
        Waiting,
        Prefill,
        Decode,
        Finished,
    };

    uint64_t           id    = 0;
    State              state = Waiting;

    std::vector<Token> input;        // prompt token ids（提交后不可变）
    std::vector<Token> output;       // 已生成 token（output.size() == decode_pos + 1 in Decode）

    // 三个 scheduler 分配的"锚点"——Request 活跃期内不变。
    //   node          : radix 树里的最深命中节点（prefix cache 视角）
    //   table_idx     : page_table 的行号（KV pool 视角），kInvalid = 尚未分配
    //   radix_hit_len : prepare 时的 radix 命中 token 数。CacheManager.finished
    //                   靠它区分"已在树里（不动）"和"本次新插（写回）"两段。
    // 三者在 Waiting->Prefill 入选时一起赋值，Finished 后由 scheduler 回收。
    Node*              node          = nullptr;
    Index              table_idx     = (Index)-1;
    int                radix_hit_len = 0;

    int                prefill_pos = 0;
    int                decode_pos  = 0;

    // ---- 查询 ----
    int total_len()  const { return (int)input.size()  + (int)output.size(); }
    int cached_len() const { return prefill_pos + decode_pos; }
    int extend_len() const { return total_len() - cached_len(); }

    // ---- 状态迁移 ----
    void to_prefill() { state = Prefill;  }
    void to_decode()  { state = Decode;   }
    void finish()     { state = Finished; }

    // ---- KV 槽位提交（"刚 submit 的 forward 写了哪些 KV"）----
    // prefill forward 写完整个 input 段的 KV：
    void commit_prefill_kv() {
        vt_assert(state == Prefill, "Request::commit_prefill_kv: not in Prefill");
        prefill_pos = (int)input.size();
    }
    // decode forward 写上一次预测 token 的 KV（一个 slot）：
    void commit_decode_kv() {
        vt_assert(state == Decode, "Request::commit_decode_kv: not in Decode");
        decode_pos += 1;
    }

    // ---- 预测 token 记录（"刚 submit 的 forward 预测了什么"）----
    // 把预测的 token 追加到 output；若是 prefill 完成则状态切到 Decode。
    // 调用方永远不直接 push output —— output 长度增长只走这条路径。
    void record_predicted(Token predicted) {
        vt_assert(state == Prefill || state == Decode,
                  "Request::record_predicted: state must be Prefill or Decode");
        if (state == Prefill) state = Decode;
        output.push_back(predicted);
    }

    // ---- sync 路径便捷封装：commit + record 一步到位 ----
    void on_prefill_done(Token predicted) {
        commit_prefill_kv();
        record_predicted(predicted);
    }
    void on_decode_done(Token predicted) {
        commit_decode_kv();
        record_predicted(predicted);
    }
};

} // namespace vt

#endif
