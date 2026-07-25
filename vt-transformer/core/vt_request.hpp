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
//   decode_pos  : output 中已落 KV 的 token 数
//   cached_len  = prefill_pos + decode_pos
//   total_len   = input.size() + output.size()
//   extend_len  = total_len - cached_len   （本次 forward 待写入的量）
//
// 状态迁移由 scheduler 显式驱动；Request 自身只负责薄记。
struct Request {
    enum State {
        Waiting,
        Prefill,
        Decode,
        Finished,
    };

    uint64_t           id   = 0;
    State              state = Waiting;

    std::vector<int>   input;        // prompt token ids（提交后不可变）
    std::vector<int>   output;       // 已生成 token（Decode 期间单调增长）

    // 两个 scheduler 分配的"锚点"——Request 活跃期内不变。
    //   node      : radix 树里的最深命中节点（prefix cache 视角）
    //   table_idx : page_table 的行号（KV pool 视角），-1 = 尚未分配
    // 二者在 Waiting->Prefill 入选时一起赋值，Finished 后由 scheduler 回收。
    RadixNode<>*       node      = nullptr;
    int                table_idx = -1;

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

    // Decode 一步：追加新采样的 token，output 长度自动 +1。
    // 调用方永远不直接 push output —— 长度增长只走这条路径。
    void decode_step(int token) {
        vt_assert(state == Decode, "Request::decode_step: not in Decode");
        output.push_back(token);
        decode_pos += 1;
    }
};

} // namespace vt

#endif
