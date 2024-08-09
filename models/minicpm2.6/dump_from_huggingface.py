from transformers import AutoModelForCausalLM, AutoTokenizer

'''
  (llm): Qwen2ForCausalLM(
    (model): Qwen2Model(
      (embed_tokens): Embedding(151666, 3584)
      (layers): ModuleList(
        (0-27): 28 x Qwen2DecoderLayer(
          (self_attn): Qwen2SdpaAttention(
            (q_proj): Linear(in_features=3584, out_features=3584, bias=True)
            (k_proj): Linear(in_features=3584, out_features=512, bias=True)
            (v_proj): Linear(in_features=3584, out_features=512, bias=True)
            (o_proj): Linear(in_features=3584, out_features=3584, bias=False)
            (rotary_emb): Qwen2RotaryEmbedding()
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=3584, out_features=18944, bias=False)
            (up_proj): Linear(in_features=3584, out_features=18944, bias=False)
            (down_proj): Linear(in_features=18944, out_features=3584, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
          (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
        )
      )
      (norm): Qwen2RMSNorm((3584,), eps=1e-06)
    )
    (lm_head): Linear(in_features=3584, out_features=151666, bias=False)
  )
  (vpm): SiglipVisionTransformer(
    (embeddings): SiglipVisionEmbeddings(
      (patch_embedding): Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding=valid)
      (position_embedding): Embedding(4900, 1152)
    )
    (encoder): SiglipEncoder(
      (layers): ModuleList(
        (0-26): 27 x SiglipEncoderLayer(
          (self_attn): SiglipAttention(
            (k_proj): Linear(in_features=1152, out_features=1152, bias=True)
            (v_proj): Linear(in_features=1152, out_features=1152, bias=True)
            (q_proj): Linear(in_features=1152, out_features=1152, bias=True)
            (out_proj): Linear(in_features=1152, out_features=1152, bias=True)
          )
          (layer_norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
          (mlp): SiglipMLP(
            (activation_fn): PytorchGELUTanh()
            (fc1): Linear(in_features=1152, out_features=4304, bias=True)
            (fc2): Linear(in_features=4304, out_features=1152, bias=True)
          )
          (layer_norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
        )
      )
    )
    (post_layernorm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
  )
  (resampler): Resampler(
    (kv_proj): Linear(in_features=1152, out_features=3584, bias=False)
    (attn): MultiheadAttention(
      (out_proj): Linear(in_features=3584, out_features=3584, bias=True)
    )
    (ln_q): LayerNorm((3584,), eps=1e-06, elementwise_affine=True)
    (ln_kv): LayerNorm((3584,), eps=1e-06, elementwise_affine=True)
    (ln_post): LayerNorm((3584,), eps=1e-06, elementwise_affine=True)
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

def save_llm(llm):
    ## wte & lm_head & ln_f
    w = llm.model.embed_tokens.weight
    save_weight(w, "wte");
    w = llm.model.norm.weight
    save_weight(w, "ln_f");
    w = llm.lm_head.weight
    save_weight(w, "lm_head");

    for i in range(0, 28):
        pname = "h_" + str(i) + ".";

        w = llm.model.layers[i].input_layernorm.weight;
        name = pname + "ln_1.weight";
        save_weight(w, name);

        w = llm.model.layers[i].post_attention_layernorm.weight;
        name = pname + "ln_2.weight";
        save_weight(w, name);

        w = llm.model.layers[i].self_attn.q_proj.weight;
        name = pname + "attn.query.weight";
        save_weight(w, name);
        w = llm.model.layers[i].self_attn.q_proj.bias;
        name = pname + "attn.query.bias";
        save_weight(w, name);
        w = llm.model.layers[i].self_attn.k_proj.weight;
        name = pname + "attn.key.weight";
        save_weight(w, name);
        w = llm.model.layers[i].self_attn.k_proj.bias;
        name = pname + "attn.key.bias";
        save_weight(w, name);
        w = llm.model.layers[i].self_attn.v_proj.weight;
        name = pname + "attn.value.weight";
        save_weight(w, name);
        w = llm.model.layers[i].self_attn.v_proj.bias;
        name = pname + "attn.value.bias";
        save_weight(w, name);

        w = llm.model.layers[i].self_attn.o_proj.weight;
        name = pname + "attn.o_proj.weight";
        save_weight(w, name);

        w = llm.model.layers[i].mlp.gate_proj.weight;
        name = pname + "mlp.w1.weight";
        save_weight(w, name);

        w = llm.model.layers[i].mlp.up_proj.weight;
        name = pname + "mlp.w2.weight";
        save_weight(w, name);

        w = llm.model.layers[i].mlp.down_proj.weight;
        name = pname + "mlp.o_proj.weight";
        save_weight(w, name);

pretrain = "./";  ## "openbmb/MiniCPM-V-2_6"
model = AutoModelForCausalLM.from_pretrained(pretrain, device_map="cpu", trust_remote_code=True).eval()

save_llm(model.llm);



