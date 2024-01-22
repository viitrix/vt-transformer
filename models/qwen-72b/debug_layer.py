from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tokenization_qwen import QWenTokenizer

tokenizer = QWenTokenizer("./qwen.tiktoken");
model = AutoModelForCausalLM.from_pretrained("./", device_map="cpu", trust_remote_code=True, bf16=True).eval()

response, history = model.chat(tokenizer, "hello world", history=None)

