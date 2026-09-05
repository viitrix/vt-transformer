#!/usr/bin/env python
# encoding: utf-8

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Dict, List, Literal, Optional, Tuple

import zmq
import zmq.asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer


# 全局相关配置：
#   req 通道: frontend PUSH -> backend PULL (generate / abort)
#   out 通道: backend PUSH -> frontend PULL (output / error, 按 rid 路由)
REQ_SOCK_PATH = "ipc:///tmp/vtt_req.sock"
OUT_SOCK_PATH = "ipc:///tmp/vtt_out.sock"
LLM_MODEL_PATH = "../../qwen3-0.6b"
HTTP_API_PORT = 8088


# ==========================================
# 1. OpenAI API 数据结构定义 (Pydantic Schema)
# ==========================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-0.6b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]

class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vt-transformer"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ==========================================
# 2. ZMQ 后端通信层 (JSON over ipc://)
# ==========================================

class BackendClient:
    """与 C++ 推理后端通信：发送 generate/abort 请求，按 rid 接收流式回传。"""

    def __init__(self):
        ctx = zmq.asyncio.Context()
        self.req_sock = ctx.socket(zmq.PUSH)
        self.req_sock.connect(REQ_SOCK_PATH)
        self.out_sock = ctx.socket(zmq.PULL)
        self.out_sock.connect(OUT_SOCK_PATH)
        self.rid_counter = 0
        self.pending: Dict[int, asyncio.Queue] = {}
        self._recv_task = None

    async def start(self):
        # 保存引用，避免任务被 GC 回收
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self):
        while True:
            msg = json.loads(await self.out_sock.recv_string())
            q = self.pending.get(msg.get("rid"))
            if q is not None:
                await q.put(msg)

    async def generate(self, input_ids: List[int], sampling: dict, stream: bool) -> Tuple[int, asyncio.Queue]:
        self.rid_counter += 1
        rid = self.rid_counter
        q: asyncio.Queue = asyncio.Queue()
        self.pending[rid] = q
        await self.req_sock.send_string(json.dumps({
            "type": "generate",
            "rid": rid,
            "input_ids": input_ids,
            "sampling": sampling,
            "stream": stream,
        }))
        return rid, q

    async def abort(self, rid: int):
        await self.req_sock.send_string(json.dumps({"type": "abort", "rid": rid}))
        self.pending.pop(rid, None)

    def release(self, rid: int):
        self.pending.pop(rid, None)


class IncrementalDetokenizer:
    """每步对完整序列重新 decode 再取新增后缀，跨 token 的多字节字符会在补齐后一次性输出。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: List[int] = []
        self.sent_len = 0

    def next_text(self, new_ids: List[int]) -> str:
        self.ids.extend(new_ids)
        full = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        delta = full[self.sent_len:]
        self.sent_len = len(full)
        return delta


# ==========================================
# 3. FastAPI 应用与服务逻辑
# ==========================================

app = FastAPI(title="Lite OpenAI Compatible Serving")

backend = BackendClient()
tokenizer = None


@app.on_event("startup")
async def startup_event():
    global tokenizer
    await backend.start()

    model_name_or_path = LLM_MODEL_PATH
    print(f"[Server] Loading Tokenizer from {model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        print("[Server] Tokenizer Loaded Successfully!")
    except Exception as e:
        print(f"[Warning] Failed to load tokenizer ({e}), fallback will be used.")


@app.get("/v1/models")
async def list_models():
    return ModelListResponse(data=[ModelInfo(id="qwen3-0.6b", created=int(time.time()))])


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer is not initialized.")

    # 1. 应用 Chat Template，直接得到 token ids
    try:
        input_ids = tokenizer.apply_chat_template(
            [m.model_dump() for m in request.messages],
            tokenize=True,
            add_generation_prompt=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to apply chat template: {str(e)}")

    # 2. 提交到 C++ 后端
    sampling = {
        "temperature": request.temperature if request.temperature is not None else 0.7,
        "top_p": request.top_p if request.top_p is not None else 1.0,
        "top_k": 0,
        "max_new_tokens": request.max_tokens or 128,
        "stop_token_ids": [tokenizer.eos_token_id],
    }
    rid, response_queue = await backend.generate(input_ids, sampling, bool(request.stream))
    print(f"[Server] rid={rid} input_tokens={len(input_ids)} stream={bool(request.stream)}")
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # 3. 流式响应 (stream=True) -> SSE 逐 Token 返回
    if request.stream:
        async def stream_generator() -> AsyncGenerator[str, None]:
            detok = IncrementalDetokenizer(tokenizer)
            finished = False
            try:
                # 3.1 首个包发送 role: assistant
                first_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created_time,
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(role="assistant"),
                        finish_reason=None
                    )]
                )
                yield f"data: {first_chunk.model_dump_json()}\n\n"

                # 3.2 循环从 ZMQ 接收 output 消息
                while True:
                    msg = await response_queue.get()
                    if msg["type"] == "error":
                        print(f"[Engine] rid={rid} error: {msg.get('message')}")
                        break

                    text = detok.next_text(msg.get("tokens", []))
                    if text:
                        chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created_time,
                            model=request.model,
                            choices=[ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(content=text),
                                finish_reason=None
                            )]
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                    finish = msg.get("finish")
                    if finish is not None:
                        finished = True
                        final_chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created_time,
                            model=request.model,
                            choices=[ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(),
                                finish_reason="length" if finish == "length" else "stop"
                            )]
                        )
                        yield f"data: {final_chunk.model_dump_json()}\n\n"
                        break

                # 3.3 遵循 OpenAI 规范，以 [DONE] 结尾
                yield "data: [DONE]\n\n"
            finally:
                # 客户端提前断开 (generator 被 cancel) 时通知后端取消该请求
                if finished:
                    backend.release(rid)
                else:
                    await backend.abort(rid)

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # 4. 非流式响应 (stream=False) -> 收齐所有 Token 后一次性返回
    else:
        all_ids: List[int] = []
        finish_reason = "stop"
        finished = False
        try:
            while True:
                msg = await response_queue.get()
                if msg["type"] == "error":
                    raise HTTPException(status_code=500, detail=str(msg.get("message", "backend error")))
                all_ids.extend(msg.get("tokens", []))
                finish = msg.get("finish")
                if finish is not None:
                    finished = True
                    if finish == "length":
                        finish_reason = "length"
                    break
        finally:
            if finished:
                backend.release(rid)
            else:
                await backend.abort(rid)

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=tokenizer.decode(all_ids, skip_special_tokens=True)),
                finish_reason=finish_reason
            )]
        )
        return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=HTTP_API_PORT)
