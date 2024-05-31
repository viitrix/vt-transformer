from transformers import AutoModelForCausalLM, AutoTokenizer

'''
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
'''

def save_weight(w, wfile):
    print("dumping " + wfile + " ...");
    ofile_name = "./weights/" + wfile + ".fp16";
    w16 = w.cpu().float().detach().numpy().flatten().astype('float16');
    w16.tofile(ofile_name);

    '''
    ofile_name = "./weights/" + wfile + ".fp32";
    w32 = w.cpu().float().detach().numpy().flatten();
    w32.tofile(ofile_name);
    '''

pretrain = "./";  ## shenzhi-wang/Llama3-8B-Chinese-Chat
model = AutoModelForCausalLM.from_pretrained(pretrain, device_map="cpu").eval()

## wte & lm_head & ln_f
w = model.model.embed_tokens.weight
save_weight(w, "wte");
w = model.model.norm.weight
save_weight(w, "ln_f");
w = model.lm_head.weight
save_weight(w, "lm_head");

for i in range(0, 32):
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
    w = model.model.layers[i].self_attn.k_proj.weight;
    name = pname + "attn.key.weight";
    save_weight(w, name);
    w = model.model.layers[i].self_attn.v_proj.weight;
    name = pname + "attn.value.weight";
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
