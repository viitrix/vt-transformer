import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

## '.' -> 'openbmb/MiniCPM-V-2_6'
model = AutoModel.from_pretrained('.', trust_remote_code=True, torch_dtype=torch.float16,
        attn_implementation='sdpa') # sdpa or flash_attention_2, no eager
tokenizer = AutoTokenizer.from_pretrained('.', trust_remote_code=True)

## debug code in cuda mode
model = model.eval().cuda()

## prepare chat message
image = Image.open('./x.jpg').convert('RGB')
question = '描述一下画面.'
msgs = [{'role': 'user', 'content': [image, question]}]

## just forward
res = model.chat( image=None, msgs=msgs, tokenizer=tokenizer)

print(res);
