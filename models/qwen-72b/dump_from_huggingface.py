from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


'''
QWenLMHeadModel(
  (transformer): QWenModel(
    (wte): Embedding(152064, 8192)
    (drop): Dropout(p=0.0, inplace=False)
    (rotary_emb): RotaryEmbedding()
    (h): ModuleList(
      (0-79): 80 x QWenBlock(
        (ln_1): RMSNorm()
        (attn): QWenAttention(
          (c_attn): Linear(in_features=8192, out_features=24576, bias=True)
          (c_proj): Linear(in_features=8192, out_features=8192, bias=False)
          (attn_dropout): Dropout(p=0.0, inplace=False)
        )
        (ln_2): RMSNorm()
        (mlp): QWenMLP(
          (w1): Linear(in_features=8192, out_features=24576, bias=False)
          (w2): Linear(in_features=8192, out_features=24576, bias=False)
          (c_proj): Linear(in_features=24576, out_features=8192, bias=False)
        )
      )
    )
   (ln_f): RMSNorm()
  )
  (lm_head): Linear(in_features=8192, out_features=152064, bias=False)
)
'''

def save_weight(w, wfile):
    ofile_name = "./weights/" + wfile + ".fp16";
    print("dumping " + ofile_name + " ...");
    w16 = w.cpu().float().detach().numpy().flatten().astype('float16');
    w16.tofile(ofile_name);

pretrain = "./";  ## "Qwen/Qwen-72B-Chat"
model = AutoModelForCausalLM.from_pretrained(pretrain, device_map="cpu", trust_remote_code=True).eval()

## wte & lm_head & ln_f
w = model.transformer.wte.weight
save_weight(w, "wte");
w = model.transformer.ln_f.weight
save_weight(w, "ln_f");
w = model.lm_head.weight
save_weight(w, "lm_head");

for i in range(0, 80):
    pname = "h_" + str(i) + ".";

    w = model.transformer.h[i].ln_1.weight;
    name = pname + "ln_1.weight";
    save_weight(w, name);

    w = model.transformer.h[i].ln_2.weight;
    name = pname + "ln_2.weight";
    save_weight(w, name);

    w = model.transformer.h[i].attn.c_attn.weight;
    name = pname + "attn.c_attn.weight";
    save_weight(w, name);
    w = model.transformer.h[i].attn.c_attn.bias;
    name = pname + "attn.c_attn.bias";
    save_weight(w, name);

    w = model.transformer.h[i].attn.c_proj.weight;
    name = pname + "attn.c_proj.weight";
    save_weight(w, name);

    w = model.transformer.h[i].mlp.w1.weight;
    name = pname + "mlp.w1.weight";
    save_weight(w, name);

    w = model.transformer.h[i].mlp.w2.weight;
    name = pname + "mlp.w2.weight";
    save_weight(w, name);

    w = model.transformer.h[i].mlp.c_proj.weight;
    name = pname + "mlp.c_proj.weight";
    save_weight(w, name);

