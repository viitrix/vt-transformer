#!/usr/bin/env python
# encoding: utf-8
#
# 推理过程调试 CLI：走与 http_main.py 相同的 ZMQ 通道与消息协议，
# 直接在终端流式打印生成结果，并输出 TTFT / 吞吐等计时信息。
#
# 注意: out 通道的消息会被任意一个已连接的前端抢走，
# 调试时不要与 http_main.py 同时连接同一个后端。
#
# 用法:
#   python cli_main.py                          # 交互式多轮对话
#   python cli_main.py "你好，介绍一下自己"       # 单次提问
#   python cli_main.py -v "1+1=?"               # 打印每步原始消息与 token ids

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

import zmq
from transformers import AutoTokenizer

REQ_SOCK_PATH = "ipc:///tmp/vtt_req.sock"
OUT_SOCK_PATH = "ipc:///tmp/vtt_out.sock"
LLM_MODEL_PATH = "/home/teaonly/workspace/qwen3-0.6b"

RECV_TIMEOUT_S = 120.0


class BackendClient:
    """同步 ZMQ 客户端: 发送 generate/abort，阻塞等待按 rid 路由的回传。"""

    def __init__(self):
        ctx = zmq.Context.instance()
        self.req_sock = ctx.socket(zmq.PUSH)
        self.req_sock.connect(REQ_SOCK_PATH)
        self.out_sock = ctx.socket(zmq.PULL)
        self.out_sock.connect(OUT_SOCK_PATH)
        self.out_sock.setsockopt(zmq.RCVTIMEO, 500)  # 轮询式阻塞，保证 Ctrl+C 可中断
        self.rid_counter = 0

    def generate(self, input_ids: List[int], sampling: dict, stream: bool) -> int:
        self.rid_counter += 1
        rid = self.rid_counter
        self.req_sock.send_string(json.dumps({
            "type": "generate",
            "rid": rid,
            "input_ids": input_ids,
            "sampling": sampling,
            "stream": stream,
        }))
        return rid

    def abort(self, rid: int):
        self.req_sock.send_string(json.dumps({"type": "abort", "rid": rid}))

    def next_msg(self, rid: int) -> Optional[dict]:
        """取下一条属于 rid 的消息；超时返回 None。"""
        deadline = time.monotonic() + RECV_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                raw = self.out_sock.recv_string()
            except zmq.Again:
                continue
            msg = json.loads(raw)
            if msg.get("rid") == rid:
                return msg
            # 丢弃路由到别的 rid 的残留消息
        return None


class IncrementalDetokenizer:
    """每步对完整序列重新 decode 再取新增后缀。多字节字符被 token 切开时，
    中间态 decode 会在尾部产生 U+FFFD，此时先扣住不发（sent_len 不推进），
    待字符在后续 token 补齐后一并输出。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: List[int] = []
        self.sent_len = 0

    def next_text(self, new_ids: List[int]) -> str:
        self.ids.extend(new_ids)
        full = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if full.endswith("�"):
            full = full[:-1]
        delta = full[self.sent_len:]
        self.sent_len = len(full)
        return delta


def run_request(
    backend: BackendClient,
    tokenizer,
    input_ids: List[int],
    sampling: dict,
    verbose: bool,
) -> Tuple[str, List[int]]:
    """执行一次完整生成，终端流式打印，返回 (finish_reason, all_token_ids)。"""
    rid = backend.generate(input_ids, sampling, stream=True)
    n_in = len(input_ids)
    print(f"[cli] rid={rid} input_tokens={n_in} "
          f"max_new_tokens={sampling['max_new_tokens']} "
          f"temperature={sampling['temperature']} top_p={sampling['top_p']}")

    detok = IncrementalDetokenizer(tokenizer)
    all_ids: List[int] = []
    finish_reason = "error"
    t_send = time.monotonic()
    ttft = None
    n_steps = 0

    print("[assistant] ", end="", flush=True)
    while True:
        msg = backend.next_msg(rid)
        if msg is None:
            print(f"\n[cli] 超过 {RECV_TIMEOUT_S:.0f}s 未收到后端消息，中止本次请求")
            backend.abort(rid)
            break

        if verbose:
            print(f"\n[recv] {json.dumps(msg, ensure_ascii=False)}")

        if msg["type"] == "error":
            print(f"\n[engine] error: {msg.get('message')}")
            backend.abort(rid)
            break

        if ttft is None:
            ttft = time.monotonic() - t_send

        tokens = msg.get("tokens", [])
        all_ids.extend(tokens)
        n_steps += 1
        text = detok.next_text(tokens)
        if text:
            print(text, end="", flush=True)

        finish = msg.get("finish")
        if finish is not None:
            finish_reason = finish
            break

    elapsed = time.monotonic() - t_send
    if ttft is not None and all_ids:
        decode_elapsed = max(elapsed - ttft, 1e-6)
        print(f"\n[stats] finish={finish_reason} new_tokens={len(all_ids)} steps={n_steps} "
              f"ttft={ttft:.3f}s total={elapsed:.3f}s "
              f"decode={len(all_ids) / decode_elapsed:.2f} tok/s")
    else:
        print(f"\n[stats] finish={finish_reason} new_tokens={len(all_ids)} total={elapsed:.3f}s")
    return finish_reason, all_ids


def build_sampling(args) -> dict:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": 0,
        "max_new_tokens": args.max_tokens,
        "stop_token_ids": [],  # 由 main() 填入 eos
    }


def interactive_loop(backend: BackendClient, tokenizer, args):
    messages: List[Dict[str, str]] = []
    print("交互模式: 输入内容对话，/clear 重置上下文，/exit 或 Ctrl+C 退出\n")
    while True:
        try:
            user = input("[user] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/clear":
            messages.clear()
            print("[cli] 上下文已重置\n")
            continue

        messages.append({"role": "user", "content": user})
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        sampling = build_sampling(args)
        sampling["stop_token_ids"] = [tokenizer.eos_token_id]

        finish_reason, out_ids = run_request(backend, tokenizer, input_ids, sampling, args.verbose)
        messages.append({
            "role": "assistant",
            "content": tokenizer.decode(out_ids, skip_special_tokens=True),
        })
        print()


def main():
    parser = argparse.ArgumentParser(description="vt-transformer 推理调试 CLI")
    parser.add_argument("prompt", nargs="*", help="单次提问内容；留空则进入交互模式")
    parser.add_argument("--model", default=LLM_MODEL_PATH, help="tokenizer/模型目录")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("-v", "--verbose", action="store_true", help="打印每步原始 ZMQ 消息")
    args = parser.parse_args()

    print(f"[cli] Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    backend = BackendClient()
    print(f"[cli] connected: req={REQ_SOCK_PATH} out={OUT_SOCK_PATH}")

    prompt = " ".join(args.prompt).strip()
    if prompt:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=False
        )
        sampling = build_sampling(args)
        sampling["stop_token_ids"] = [tokenizer.eos_token_id]
        run_request(backend, tokenizer, input_ids, sampling, args.verbose)
    else:
        interactive_loop(backend, tokenizer, args)


if __name__ == "__main__":
    main()
