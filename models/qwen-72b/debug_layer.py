from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tokenization_qwen import QWenTokenizer

target = None;

tokenizer = QWenTokenizer("./qwen.tiktoken");
model = AutoModelForCausalLM.from_pretrained("./", device_map="auto", trust_remote_code=True).eval()
model = model.half();

response, history = model.chat(tokenizer, "hello world in the sky", history=None)
