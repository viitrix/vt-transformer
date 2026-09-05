#!/usr/bin/env python
# encoding: utf-8

import torchvision
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

'''
image = Image.open('./demo.jpg');
image = image.convert("RGB");
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
image_preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize( (448, 448), interpolation=torchvision.transforms.InterpolationMode.BICUBIC ),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean, std=std)
]);
result = image_preprocess(image);
'''


# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./", device_map="cuda", trust_remote_code=True, fp16=True).eval()

### debug language part
#response, history = model.chat(tokenizer, "hello world in the sky", history=None)

### debug visual part
#vfeature = model.transformer.visual.encode(["./demo.png"]);

query = tokenizer.from_list_format([
        {'image': './demo.png'},
        {'text': '图片上的道路机动车是否可以通行？'},
        ])
print(query);

#response, history = model.chat(tokenizer, query=query, history=None)
#print(response)

