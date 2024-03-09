from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from torch.nn import functional as F
import math
import torch

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos


'''
QWenLMHeadModel(
  (transformer): QWenModel(
    (wte): Embedding(151936, 4096)
    (drop): Dropout(p=0.0, inplace=False)
    (rotary_emb): RotaryEmbedding()
    (h): ModuleList(
      (0-31): 32 x QWenBlock(
        (ln_1): RMSNorm()
        (attn): QWenAttention(
          (c_attn): Linear(in_features=4096, out_features=12288, bias=True)
          (c_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (attn_dropout): Dropout(p=0.0, inplace=False)
        )
        (ln_2): RMSNorm()
        (mlp): QWenMLP(
          (w1): Linear(in_features=4096, out_features=11008, bias=False)
          (w2): Linear(in_features=4096, out_features=11008, bias=False)
          (c_proj): Linear(in_features=11008, out_features=4096, bias=False)
        )
      )
    )
    (ln_f): RMSNorm()
    (visual): VisionTransformer(
      (conv1): Conv2d(3, 1664, kernel_size=(14, 14), stride=(14, 14), bias=False)
      (ln_pre): LayerNorm((1664,), eps=1e-06, elementwise_affine=True)
      (transformer): TransformerBlock(
        (resblocks): ModuleList(
          (0-47): 48 x VisualAttentionBlock(
            (ln_1): LayerNorm((1664,), eps=1e-06, elementwise_affine=True)
            (ln_2): LayerNorm((1664,), eps=1e-06, elementwise_affine=True)
            (attn): VisualAttention(
              (in_proj): Linear(in_features=1664, out_features=4992, bias=True)
              (out_proj): Linear(in_features=1664, out_features=1664, bias=True)
            )
            (mlp): Sequential(
              (c_fc): Linear(in_features=1664, out_features=8192, bias=True)
              (gelu): GELU(approximate='none')
              (c_proj): Linear(in_features=8192, out_features=1664, bias=True)
            )
          )
        )
      )
      (attn_pool): Resampler(
        (kv_proj): Linear(in_features=1664, out_features=4096, bias=False)
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=4096, out_features=4096, bias=True)
        )
        (ln_q): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
        (ln_kv): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
      )
      (ln_post): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
    )
  )
  (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
)
'''

def save_weight(w, wfile, f32=False):
    print("dumping " + wfile + " ...");
    ofile_name = "./weights/" + wfile + ".fp16";
    w16 = w.cpu().float().detach().numpy().flatten().astype('float16');
    w16.tofile(ofile_name);

    if ( f32 ):
        ofile_name = "./weights/" + wfile + ".fp32";
        w32 = w.cpu().float().detach().numpy().flatten();
        w32.tofile(ofile_name);


feature_size = 4096;
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./", device_map="cuda", trust_remote_code=True).eval()

## wte & lm_head & ln_f
w = model.transformer.wte.weight
save_weight(w, "wte");
w = model.transformer.ln_f.weight
save_weight(w, "ln_f");
w = model.lm_head.weight
save_weight(w, "lm_head");

for i in range(0, 32):
    pname = "h_" + str(i) + ".";

    w = model.transformer.h[i].ln_1.weight;
    name = pname + "ln_1.weight";
    save_weight(w, name);

    w = model.transformer.h[i].ln_2.weight;
    name = pname + "ln_2.weight";
    save_weight(w, name);

    w = model.transformer.h[i].attn.c_attn.weight;
    [q,k,v] = torch.split(w, feature_size);
    name = pname + "attn.query.weight";
    save_weight(q, name);
    name = pname + "attn.key.weight";
    save_weight(k, name);
    name = pname + "attn.value.weight";
    save_weight(v, name);

    w = model.transformer.h[i].attn.c_attn.bias;
    [q,k,v] = torch.split(w, feature_size);
    name = pname + "attn.query.bias";
    save_weight(q, name);
    name = pname + "attn.key.bias";
    save_weight(k, name);
    name = pname + "attn.value.bias";
    save_weight(v, name);

    w = model.transformer.h[i].attn.c_proj.weight;
    name = pname + "attn.o_proj.weight";
    save_weight(w, name);

    w = model.transformer.h[i].mlp.w1.weight;
    name = pname + "mlp.w1.weight";
    save_weight(w, name);

    w = model.transformer.h[i].mlp.w2.weight;
    name = pname + "mlp.w2.weight";
    save_weight(w, name);

    w = model.transformer.h[i].mlp.c_proj.weight;
    name = pname + "mlp.o_proj.weight";
    save_weight(w, name);


### visual part of Qwen-vl
w = model.transformer.visual.conv1.weight
save_weight(w, "v.conv1.weight", True);

w = get_abs_pos( model.transformer.visual.positional_embedding, 1024);
save_weight(w, "v.pos_emb");
w = model.transformer.visual.conv1.weight
save_weight(w, "v.conv1.weight");
w = model.transformer.visual.ln_pre.weight
save_weight(w, "v.ln_pre.weight");
w = model.transformer.visual.ln_pre.bias
save_weight(w, "v.ln_pre.bias");
w = model.transformer.visual.ln_post.weight
save_weight(w, "v.ln_post.weight");
w = model.transformer.visual.ln_post.bias
save_weight(w, "v.ln_post.bias");

w = model.transformer.visual.attn_pool.kv_proj.weight
save_weight(w, "v.pool.kv_proj.weight");
w = model.transformer.visual.attn_pool.attn.out_proj.weight
save_weight(w, "v.pool.out_proj.weight");
w = model.transformer.visual.attn_pool.attn.out_proj.bias
save_weight(w, "v.pool.out_proj.bias");
w = model.transformer.visual.attn_pool.ln_kv.weight
save_weight(w, "v.pool.ln_kv.weight");
w = model.transformer.visual.attn_pool.ln_kv.bias
save_weight(w, "v.pool.ln_kv.bias");


"""
w = model.transformer.visual.attn_pool.pos_embed
save_weight(w, "v.pool.pos_emb");
w = model.transformer.visual.attn_pool.ln_q.weight
save_weight(w, "v.pool.ln_q.weight");
w = model.transformer.visual.attn_pool.ln_q.bias
save_weight(w, "v.pool.ln_q.bias");
"""

w = model.transformer.visual.attn_pool.ln_q(model.transformer.visual.attn_pool.query);
w = w + model.transformer.visual.attn_pool.pos_embed
save_weight(w, "v.pool.query");
w = get_abs_pos(model.transformer.visual.attn_pool.pos_embed, 1024);
save_weight(w, "v.pool.pos_emb");

raise Exception("==DEBUG==");

blocks = model.transformer.visual.transformer.resblocks;
for i in range(0, 48):
    pname = "v.b_" + str(i) + ".";

    w = blocks[i].ln_1.weight;
    name = pname + "ln_1.weight"
    save_weight(w, name);
    w = blocks[i].ln_1.bias;
    name = pname + "ln_1.bias"
    save_weight(w, name);

    w = blocks[i].ln_2.weight;
    name = pname + "ln_2.weight"
    save_weight(w, name);
    w = blocks[i].ln_2.bias;
    name = pname + "ln_2.bias"
    save_weight(w, name);

    w = blocks[i].attn.in_proj.weight;
    [wq, wk, wv] = w.view([16, 104*3, 1664]).split(104, 1);
    save_weight(wq, pname + "attn.in_proj_q.weight");
    save_weight(wk, pname + "attn.in_proj_k.weight");
    save_weight(wv, pname + "attn.in_proj_v.weight");

    w = blocks[i].attn.in_proj.bias;
    [bq, bk, bv] = w.view([16, 104*3]).split(104, 1);
    save_weight(bq, pname + "attn.in_proj_q.bias");
    save_weight(bk, pname + "attn.in_proj_k.bias");
    save_weight(bv, pname + "attn.in_proj_v.bias");

    w = blocks[i].attn.out_proj.weight;
    name = pname + "attn.out_proj.weight";
    save_weight(w, name);
    w = blocks[i].attn.out_proj.bias;
    name = pname + "attn.out_proj.bias";
    save_weight(w, name);

    w = blocks[i].mlp.c_fc.weight;
    name = pname + "mlp.c_fc.weight";
    save_weight(w, name);
    w = blocks[i].mlp.c_fc.bias;
    name = pname + "mlp.c_fc.bias";
    save_weight(w, name);
    w = blocks[i].mlp.c_proj.weight;
    name = pname + "mlp.c_proj.weight";
    save_weight(w, name);
    w = blocks[i].mlp.c_proj.bias;
    name = pname + "mlp.c_proj.bias";
    save_weight(w, name);

