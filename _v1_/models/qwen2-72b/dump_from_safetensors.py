#!/usr/bin/env python
# encoding: utf-8

import json
from safetensors import safe_open

maping_file = "dump_maping.json";
repo_path = "/home/teaonly/workspace/LLMs/Qwen2-72B-Instruct";

def convert_name(tname):
    tname = tname.replace("lm_head.weight", "lm_head");
    tname = tname.replace("model.norm.weight", "ln_f");
    tname = tname.replace("model.embed_tokens.weight", "wte");

    tname = tname.replace("model.layers.", "h_");

    tname = tname.replace("self_attn", "attn");
    tname = tname.replace("k_proj", "key");
    tname = tname.replace("q_proj", "query");
    tname = tname.replace("v_proj", "value");

    tname = tname.replace("input_layernorm.weight", "ln1");
    tname = tname.replace("post_attention_layernorm.weight", "ln2");
    tname = tname.replace("gate_proj", "w1");
    tname = tname.replace("up_proj", "w2");
    tname = tname.replace("down_proj", "o_proj");

    return tname;


def save_weight(w, wfile):
    print("dumping " + wfile + " ...");
    ofile_name = "./weights/" + wfile + ".fp16";
    w16 = w.cpu().float().detach().numpy().flatten().astype('float16');
    w16.tofile(ofile_name);

for i in range(1, 38):
    ii = str(100000 + i)[1:];
    safe_name = "model-" + ii + "-of-00037.safetensors";
    safe_name = repo_path + "/" + safe_name;

    with safe_open(safe_name, framework="pt", device="cpu") as f:
        for k in f.keys():
            target = f.get_tensor(k);
            target_name =  convert_name (k);
            save_weight(target, target_name);
