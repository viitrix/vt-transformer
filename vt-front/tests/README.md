# vt-front 测试说明

## 环境准备（一次性）

```bash
cd vt-front
uv venv
uv pip install -e '.[dev]'
```

`.[dev]` 已包含全部所需：pytest、httpx（TestClient 依赖）、ruff，以及
transformers/tokenizers —— e2e 用它在本地生成微型 tokenizer，**不需要下载模型、
不需要 GPU、不需要网络**。

## 运行

```bash
# 单元 + 回归测试（三个 test_*.py，约 13s）
.venv/bin/python -m pytest tests/ -q
# 预期：57 passed

# 集成 smoke（真实 ZMQ + spawn 子进程，约 1 分钟）
.venv/bin/python tests/e2e_smoke.py
# 预期：E2E_TEST_PASSED
```

只跑某一层：

```bash
.venv/bin/python -m pytest tests/test_regressions.py      # 核心行为回归
.venv/bin/python -m pytest tests/test_failure_paths.py    # 失败路径语义
.venv/bin/python -m pytest tests/test_rereview.py         # 复审修复回归
```

## 各文件分工

| 文件 | 职责 |
|------|------|
| `test_regressions.py` | 核心行为回归；定义 `FakeTokenizer`/`FakeQueue`/`manager` 等共享基建（最底层） |
| `test_failure_paths.py` | 失败路径：错误/超时/断连/poison 消息/启动失败 → 明确的 HTTP 错误，无状态残留 |
| `test_rereview.py` | 复审修复的定点回归：发送重试、shell 容错、进程收割健壮性 |
| `e2e_smoke.py` | 端到端集成；真实 ZMQ 链路 + 双 worker 模式 + 进程/IPC 清理断言。**手动运行**，pytest 不收集 |

注意：前三个文件存在单向 import 依赖（`regressions` ← `failure_paths` ← `rereview`），
`test_regressions.py` 不能单独删除；`e2e_smoke.py` 完全独立。
