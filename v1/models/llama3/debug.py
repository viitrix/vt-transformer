#!/usr/bin/env python
# encoding: utf-8

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

model_id = "shenzhi-wang/Llama3-8B-Chinese-Chat"

tokenizer = AutoTokenizer.from_pretrained("./")
model = AutoModelForCausalLM.from_pretrained(
    "./", torch_dtype=torch.float16, device_map="auto"
)

messages = [
    {"role": "user", "content": "who are you"},
]

#
# checking chat template
talk = tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True);

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=8192,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

