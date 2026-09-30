#!/usr/bin/env python3
"""Build an FP8-dense variant of RadixArk/Qwen3.8-Flash-Next-NVFP4 (ModelOpt MIXED_PRECISION).

The published NVFP4 checkpoint quantizes only the routed experts; attention, linear attention and
the shared expert stay BF16, and on a bandwidth-bound decode those dense layers are most of the
bytes read per token. This converts exactly those projections to FP8 and leaves everything else
bit-for-bit:

  routed experts                                   NVFP4 W4A4 (untouched -> native FP4 MoE kernel)
  PLE n-gram table                                 FP8 (untouched)
  self_attn.{q,k,v,o}_proj                         -> FP8_PER_CHANNEL_PER_TOKEN
  linear_attn.{in_proj_qkv,in_proj_z,out_proj}     -> FP8_PER_CHANNEL_PER_TOKEN
  mlp.shared_expert.{gate,up,down}_proj            -> FP8_PER_CHANNEL_PER_TOKEN
  in_proj_a/b, indexer, router, norms, HC, embed, lm_head, MTP, vision   BF16 (untouched)

FP8_PER_CHANNEL_PER_TOKEN = E4M3 weight [out, in] + F32 per-output-channel weight_scale [out],
activations quantized dynamically per token -> no calibration data needed.

Only the four model-bf16-*.safetensors shards are rewritten (~5.8 GB BF16 -> ~2.9 GB FP8);
every other file is hardlinked, so <src> and <out> must be on the same filesystem. Streams one
tensor at a time and flushes ~1 GB output files, so it runs in a few GB of RAM (fine next to a
serving engine). Needs torch + safetensors -- the vLLM image has both:

  docker run --rm --user $(id -u):$(id -g) -v $HOME:/h -e HOME=/tmp --entrypoint python3 \\
    vllm/vllm-openai:nightly /h/make-fp8-dense.py \\
    /h/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/<rev> \\
    /h/models/qwen38-fn-nvfp4-fp8dense

(Mount $HOME once: hardlinks fail across two separate bind mounts even on the same filesystem.)
Deterministic: running it on each node gives byte-identical output.

usage: make-fp8-dense.py <src_snapshot> <out_dir>
"""
import json, os, re, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
SRC, OUT = sys.argv[1], sys.argv[2]
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FLUSH = 1 << 30
LM = "model.language_model.layers."
CONVERT = re.compile(
    r"^model\.language_model\.layers\.(\d+)\."
    r"(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.shared_expert\.(gate|up|down)_proj)"
    r"\.weight$")

os.makedirs(OUT, exist_ok=True)
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wm = idx["weight_map"]
dense_files = sorted({f for f in wm.values() if f.startswith("model-bf16-")})
print("rewriting", dense_files, flush=True)

t0 = time.time(); new_wm = {}; buf = {}; buf_bytes = 0; nfile = 0; converted = []
def flush():
  global buf, buf_bytes, nfile
  if not buf: return
  name = f"model-dense-{nfile:05d}.safetensors"; nfile += 1
  save_file(buf, f"{OUT}/{name}", metadata={"format": "pt"})
  for k in buf: new_wm[k] = name
  print(f"  wrote {name} {len(buf)} tensors {buf_bytes/1e9:.2f} GB {time.time()-t0:.0f}s", flush=True)
  buf = {}; buf_bytes = 0

bf16_in = fp8_out = 0
for fn in dense_files:
  with safe_open(f"{SRC}/{fn}", framework="pt") as f:
    for k in f.keys():
      t = f.get_tensor(k)
      if CONVERT.match(k):
        assert t.dtype == torch.bfloat16 and t.dim() == 2, (k, t.dtype, t.shape)
        w = t.float()
        scale = (w.abs().amax(dim=1).clamp_(min=1e-12) / FP8_MAX).to(torch.float32)
        q = (w / scale[:, None]).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        buf[k] = q; buf[k[:-len("weight")] + "weight_scale"] = scale
        bf16_in += t.numel() * 2; fp8_out += q.numel() + scale.numel() * 4
        converted.append(k[:-len(".weight")]); buf_bytes += q.numel() + scale.numel() * 4
        del w, q
      else:
        buf[k] = t.clone(); buf_bytes += t.numel() * t.element_size()
      if buf_bytes >= FLUSH: flush()
flush()
print(f"converted {len(converted)} linears: {bf16_in/1e9:.2f} GB bf16 -> {fp8_out/1e9:.2f} GB fp8", flush=True)

for k, f in wm.items():
  if f in dense_files: assert k in new_wm, k
  else: new_wm[k] = f
idx["weight_map"] = new_wm
json.dump(idx, open(f"{OUT}/model.safetensors.index.json", "w"), indent=2)

# ModelOpt MIXED_PRECISION: per-layer algo map. Anything not listed loads unquantized.
cfg = json.load(open(f"{SRC}/config.json"))
tc = cfg.get("text_config") or cfg
ql = {f"{LM}{i}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16} for i in range(tc["num_hidden_layers"])}
for p in converted:
  ql[p] = {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}
for p in sorted({k.split(".ngram_embedding.")[0] + ".ngram_embedding" for k in wm if ".ngram_embedding.shard_" in k}):
  ql[p] = {"quant_algo": "FP8"}   # PLE is also selected via text_config.ple_embedding_dtype
drop = {"*.self_attn.*", "*.linear_attn.*", "*.mlp.shared_expert.*", "*.ple.*"}
ignore = [x for x in cfg["quantization_config"]["ignore"] if x not in drop]
producer = {"name": "make-fp8-dense.py", "base": "RadixArk/Qwen3.8-Flash-Next-NVFP4"}
cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "producer": producer,
                              "kv_cache_quant_algo": None, "ignore": ignore, "quantized_layers": ql}
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=2)
json.dump({"producer": producer,
           "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None,
                            "exclude_modules": ignore, "quantized_layers": ql}},
          open(f"{OUT}/hf_quant_config.json", "w"), indent=2)
print("quantized_layers:", len(ql), "entries", flush=True)

skip = set(dense_files) | {"model.safetensors.index.json", "config.json", "hf_quant_config.json"}
for fn in os.listdir(SRC):
  if fn in skip: continue
  dst = f"{OUT}/{fn}"
  if not os.path.exists(dst): os.link(os.path.realpath(f"{SRC}/{fn}"), dst)
print("done", f"{time.time()-t0:.0f}s", flush=True)
