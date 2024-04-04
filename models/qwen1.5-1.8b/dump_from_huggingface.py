from transformers import AutoModelForCausalLM, AutoTokenizer

'''
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 2048)
    (layers): ModuleList(
      (0-23): 24 x Qwen2DecoderLayer(
        (self_attn): Qwen2SdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=2048, out_features=5504, bias=False)
          (up_proj): Linear(in_features=2048, out_features=5504, bias=False)
          (down_proj): Linear(in_features=5504, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm()
        (post_attention_layernorm): Qwen2RMSNorm()
      )
    )
    (norm): Qwen2RMSNorm()
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)
'''

def save_weight(w, wfile):
    """
    print("dumping " + wfile + " ...");
    ofile_name = "./weights/" + wfile + ".fp16";
    w16 = w.cpu().float().detach().numpy().flatten().astype('float16');
    w16.tofile(ofile_name);
    """

    ofile_name = "./weights/" + wfile + ".fp32";
    w32 = w.cpu().float().detach().numpy().flatten();
    w32.tofile(ofile_name);

pretrain = "./";  ## "Qwen/Qwen1.5-0.5B-Chat"
model = AutoModelForCausalLM.from_pretrained(pretrain, device_map="cpu").eval()

## wte & lm_head & ln_f
w = model.model.embed_tokens.weight
save_weight(w, "wte");
w = model.model.norm.weight
save_weight(w, "ln_f");
w = model.lm_head.weight
save_weight(w, "lm_head");

for i in range(0, 24):
    pname = "h_" + str(i) + ".";

    w = model.model.layers[i].input_layernorm.weight;
    name = pname + "ln_1.weight";
    save_weight(w, name);

    w = model.model.layers[i].post_attention_layernorm.weight;
    name = pname + "ln_2.weight";
    save_weight(w, name);

    w = model.model.layers[i].self_attn.q_proj.weight;
    name = pname + "attn.query.weight";
    save_weight(w, name);
    w = model.model.layers[i].self_attn.q_proj.bias;
    name = pname + "attn.query.bias";
    save_weight(w, name);
    w = model.model.layers[i].self_attn.k_proj.weight;
    name = pname + "attn.key.weight";
    save_weight(w, name);
    w = model.model.layers[i].self_attn.k_proj.bias;
    name = pname + "attn.key.bias";
    save_weight(w, name);
    w = model.model.layers[i].self_attn.v_proj.weight;
    name = pname + "attn.value.weight";
    save_weight(w, name);
    w = model.model.layers[i].self_attn.v_proj.bias;
    name = pname + "attn.value.bias";
    save_weight(w, name);

    w = model.model.layers[i].self_attn.o_proj.weight;
    name = pname + "attn.o_proj.weight";
    save_weight(w, name);

    w = model.model.layers[i].mlp.gate_proj.weight;
    name = pname + "mlp.w1.weight";
    save_weight(w, name);

    w = model.model.layers[i].mlp.up_proj.weight;
    name = pname + "mlp.w2.weight";
    save_weight(w, name);

    w = model.model.layers[i].mlp.down_proj.weight;
    name = pname + "mlp.o_proj.weight";
    save_weight(w, name);

