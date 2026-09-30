# Qwen3.8-Flash-Next NVFP4 on 2× DGX Spark — vLLM (TP2+EP, MTP3, CUDA graphs)

First working **vLLM** deployment of `RadixArk/Qwen3.8-Flash-Next-NVFP4` on **2× NVIDIA DGX Spark (GB10 / SM121)**, using the official day-0 image `vllm/vllm-openai:qwen38-flash-next` plus **one three-line patch** to the PLE quant-method resolver. Brought up 2026-08-27, the day after the model dropped.

This is the vLLM sibling of tonyd2wild's SGLang deployment
([Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)) —
same checkpoint, same hardware class, different engine. The launcher derives from
foogitiff's dual-Spark FP8 config (NVIDIA forum, Qwen3.8-Flash-Next thread, post 97),
adapted for the NVFP4 checkpoint.

> **Update 2026-09-30 — FP8-dense build, +13.5% decode, same quality.** The NVFP4 checkpoint
> quantizes only the routed experts; attention, linear attention and the shared expert stay BF16,
> and those dense layers are most of the bytes a decode step reads. `make-fp8-dense.py` converts
> just those projections to FP8 (per-channel weights, dynamic per-token activations, no calibration)
> in under a minute and leaves everything else bit-for-bit. On the same two Sparks: **56.9 → 64.6
> tok/s single stream, +15% at 172K context, HumanEval-CS 0.854 → 0.848 (a tie)**. It also runs on
> current vLLM nightlies **without the PLE patch** (fixed upstream). See
> [FP8-dense build](#fp8-dense-build-2026-09-30).

## TL;DR

1. **The official vLLM image runs on GB10** — it is multi-arch (aarch64) and registers
   `Qwen4ExpForConditionalGeneration`. No community rebuild needed.
2. **Stock, it cannot load the RadixArk NVFP4 checkpoint.** The checkpoint stores the
   51B n-gram/PLE table as FP8 shards + one global `weight_scale`, but declares `*.ple.*`
   excluded in a ModelOpt-NVFP4 quant config. vLLM's PLE resolver
   (`vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py`) only enables its FP8-PLE path
   when the *whole checkpoint* is FP8-serialized (`isinstance(quant_config, Fp8Config)`),
   so it builds a plain BF16 embedding and dies with:
   `ValueError: There is no module or parameter named 'ngram_embedding.weight_scale' ...`
3. **The image already contains everything needed** — `Qwen3_8FlashNextPLEFp8EmbeddingMethod`
   handles exactly this tensor layout (one global scale + `shard_N` FP8 tensors). The patch
   just lets it be selected under a ModelOpt-NVFP4 parent config, gated behind an env var
   (`PLE_FORCE_FP8=1`). See `ple-force-fp8.patch` — applied at runtime as a **bind-mount
   overlay**, no image rebuild.
4. NVFP4 expert kernels, the QSA path, and **full prefill + decode CUDA graph capture**
   all work on SM121 out of the box once loading succeeds.
5. **2026-09-30:** on a vLLM nightly from 2026-09-03 or later the patch is unnecessary
   ([vllm#54882](https://github.com/vllm-project/vllm/pull/54882) fixed FP8-PLE loading in mixed
   ModelOpt checkpoints) — run the launcher with `NIGHTLY=1`. Converting the BF16 dense layers to
   FP8 (`make-fp8-dense.py`) is a free **+13.5% bs1 / +22% at 8 streams**.

## Hardware

| | |
| --- | --- |
| Nodes | 2× DGX Spark (GB10, SM121), 200G ConnectX RoCE between them |
| Weights | `RadixArk/Qwen3.8-Flash-Next-NVFP4` (~126 GiB, ModelOpt NVFP4 W4A4 experts, FP8 PLE) — one copy per node (or NFS) |

## Deploy

```bash
# both nodes:
docker pull vllm/vllm-openai:qwen38-flash-next
# get ple_layer.py out of the image, apply ple-force-fp8.patch, keep it next to the launcher
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches   # unified memory: mandatory before load

# worker first (rank 1), wait ~15s, then head (rank 0):
./launch-vllm-fn.sh 1     # on the worker
./launch-vllm-fn.sh 0     # on the head — serves :8000
```

Load is ~6–7 min (206 shards; the last ~30 are the FP8 PLE shards and run slower —
that's the shard-copy doing real work, not a hang). Then warmup + graph capture, then
`/health` goes 200.

### On a current vLLM nightly (no patch)

```bash
# both nodes — pin a dated nightly tag for reproducible numbers:
docker pull vllm/vllm-openai:nightly-<sha>
NIGHTLY=1 IMAGE=vllm/vllm-openai:nightly-<sha> ./launch-vllm-fn.sh 1   # worker
NIGHTLY=1 IMAGE=vllm/vllm-openai:nightly-<sha> ./launch-vllm-fn.sh 0   # head
```

`NIGHTLY=1` drops the patch overlay and adds `--engram-config.cpu_offload false` (see gotchas).

### FP8-dense build

```bash
# on each node (deterministic, byte-identical output; ~1 min, a few GB of RAM):
docker run --rm --user $(id -u):$(id -g) -v $HOME:/h -e HOME=/tmp --entrypoint python3 \
  vllm/vllm-openai:nightly-<sha> /h/make-fp8-dense.py \
  /h/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/<rev> \
  /h/models/qwen38-fn-nvfp4-fp8dense

NIGHTLY=1 IMAGE=... MODEL=/models/qwen38-fn-nvfp4-fp8dense ./launch-vllm-fn.sh 1   # worker
NIGHTLY=1 IMAGE=... MODEL=/models/qwen38-fn-nvfp4-fp8dense ./launch-vllm-fn.sh 0   # head
```

Only the four `model-bf16-*` shards are rewritten (~5.8 GB BF16 → ~2.9 GB FP8); every other file
is hardlinked, so the output costs ~3 GB of disk. The launcher mounts `~/models` read-only at
`/models`.

## Config highlights (see launcher for the full set)

- TP2 + `--enable-expert-parallel`, `--all2all-backend allgather_reducescatter`
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (built-in MTP head)
- `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`
- `--max-num-seqs 8`, `--gpu-memory-utilization 0.80`, 262K context
- `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` all pinned to the
  inter-node interface (mind the `np0`-style suffixes on ConnectX port names)

## Benchmark

OpenAI streaming chat-completions, temp 0, greedy median of 2 reps per probe;
TTFT to first content/reasoning delta, decode = (tokens−1)/(end−first_token).
Config as in the launcher: TP2 + EP, MTP3, FULL_DECODE_ONLY CUDA graphs, seqs 8.

**Batch-1 (single stream):**

| Content    | Tokens | TTFT   | Decode tok/s |
| ---------- | ------ | ------ | ------------ |
| code       | 545    | 0.26s  | **55.8**     |
| reasoning  | 575    | 0.28s  | 56.0         |
| math       | 158    | 0.28s  | 55.6         |
| C#         | 1200   | 0.27s  | 44.4         |
| prose      | 718    | 0.31s  | 36.4         |

**Greedy median (code+reasoning+C#): 55.8 tok/s.** MTP draft acceptance ~0.50/token.

**Concurrency:**

| Streams | Per-stream median | Aggregate      | TTFT  |
| ------- | ----------------- | -------------- | ----- |
| 4       | 40.7 tok/s        | 75.8 tok/s     | 1.04s |
| 8       | 31.5 tok/s        | **126.1 tok/s**| 2.75s |

For reference, on the **same two Sparks and same checkpoint**, our best SGLang config
(day-0 image + SM121 QSA guard patch, NEXTN MTP4, CUDA graphs) measures 39.0 tok/s
bs1 / 81.2 aggregate @ 8 — the vLLM path above is ~40–55% faster across the board,
which we attribute to expert parallelism plus vLLM's MTP-in-CUDA-graphs maturity.
Numbers are day-1 kernels; no autotuning beyond the image defaults.


**Warmed steady-state (measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash), vLLM Prometheus counter sampling):**

| Streams | Aggregate    | Per-stream | TTFT  |
| ------- | ------------ | ---------- | ----- |
| 1       | 63 tok/s     | 63 tok/s   | 221ms |
| 2       | 80 tok/s     | 47 tok/s   | 368ms |
| 4       | 130 tok/s    | 37 tok/s   | 552ms |
| 8       | **203 tok/s**| 29 tok/s   | 686ms |

Two effects versus the fresh-boot table above: MTP acceptance improves as the server
warms under traffic (bs1 55.8 → 63), and counter-window sampling captures steady-state
decode without ramp-up/tail, so concurrency aggregates read higher (126 → 203 @ ×8).
Both views are honest — quote the one that matches your workload.

## FP8-dense build (2026-09-30)

`RadixArk/Qwen3.8-Flash-Next-NVFP4` keeps 300 dense projections in BF16: `self_attn.{q,k,v,o}_proj`
(12 full-attention layers), `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (36 Gated-DeltaNet
layers) and `mlp.shared_expert.{gate,up,down}_proj` (48 layers). With only 10 of 512 experts
active, those BF16 weights are most of what each decode step streams from LPDDR5x.
`make-fp8-dense.py` rewrites them as ModelOpt `FP8_PER_CHANNEL_PER_TOKEN` inside a
`MIXED_PRECISION` config:

| Component | RadixArk as shipped | FP8-dense build |
| --- | --- | --- |
| Routed experts | NVFP4 W4A4 | NVFP4 W4A4 (untouched; FlashInfer CUTLASS native FP4) |
| Attention / linear attention / shared expert | BF16 (5.82 GB) | **FP8 E4M3, per-channel scale** (2.91 GB), `CutlassFP8ScaledMMLinearKernel` |
| PLE n-gram table | FP8 | FP8 (untouched) |
| `in_proj_a/b`, indexer, router, norms, HC, embed, `lm_head`, MTP, vision | BF16 | BF16 (untouched) |

Activations are quantized dynamically per token, so there is no calibration pass. The MTP head is
left alone on purpose: draft acceptance is unchanged (mean accept length 2.89 → 2.91).

**Same two Sparks, same nightly (`nightly-ac68c308`), same flags** (`NIGHTLY=1`, seqs 8, GMU 0.80,
`AUTOTUNE=0`), clocks at boost (~2.49 GHz), 2026-09-30. Same `bench.py` probes as above, plus a
long-context probe (thinking off, 400-token answer):

| | RadixArk as shipped | FP8-dense build | Δ |
| --- | --- | --- | --- |
| bs1 greedy median (code+reasoning+C#) | 56.9 tok/s | **64.6 tok/s** | **+13.5%** |
| prose | 38.3 | 44.2 | +15% |
| C# | 47.7 | 53.3 | +12% |
| 4 streams: aggregate / per-stream | 78.0 / 32.3 | 87.8 / 37.1 | +13% / +15% |
| 8 streams: aggregate / per-stream | 118.3 / 29.9 | **144.7** / 31.1 | +22% / +4% |
| decode @ 41K / 172K context | 47.2 / 47.3 | 51.0 / **54.2** | +8% / +15% |
| prefill @ 172K | 4,270 tok/s | 4,264 tok/s | = |
| KV pool (262K ctx) | 1.98M tokens | 2.06M tokens | +4% |
| HumanEval-CS pass@1 | 0.854 | 0.848 | tie |

HumanEval-CS is MultiPL-E C#, 158 problems, greedy, thinking on, 8K budget, graded with structural
equality (so not comparable to the leaderboard, but consistent between the two columns). The two
runs disagree on 17 problems, 8 one way and 9 the other.

For reference, this recipe as originally published (day-0 image + patch) re-measures at
**59.1 tok/s bs1 / 67.5 @4 / 125.0 @8** on the same day. The nightly trades a few percent of bs1
decode for **~35–40% faster prefill** (3,155 → 4,270 tok/s at 172K).

## Gotchas that cost us time

- **Multi-pair fleets: pin NCCL to the pair's own HCA only.** If your Sparks have a second
  ConnectX port cabled to *anything else* (another pair, a transfer link), a multi-device
  `NCCL_IB_HCA` list lets vLLM's EP all2all spray traffic down it and can strangle the
  neighbouring cluster's NCCL (we measured a healthy DeepSeek pair collapse to <1 tok/s).
  Use exact-match single-device pinning: `NCCL_IB_HCA='=rocepXsYfZ'`.
- `max_tokens` includes hidden reasoning tokens (same as the SGLang lane) — budget generously.
- The vLLM image's `min_frames`/`max_frames` `[ERROR]` lines at startup are harmless
  transformers docstring lint, not failures.
- Patch placement matters: the resolver's *first* gate is the `isinstance(quant_config,
  Fp8Config)` check — an env-gated early return must go **above** it (ask us how we know).
- **Current vLLM: set PLE offload explicitly.** Nightlies no longer read `VLLM_PLE_CPU_OFFLOAD`
  (it logs "Unknown vLLM environment variable") and `EngramConfig.cpu_offload` now defaults to
  `true`. On a Spark host and GPU memory are the same LPDDR, so offload buys nothing and the pinned
  table comes out of the KV budget: **1.98M → 1.41M tokens**. Use `--engram-config.cpu_offload false`
  (`NIGHTLY=1` does this).
- **FlashInfer autotune can deadlock the two ranks.** After switching between checkpoints, the worker
  missed the autotune config cache and sat in a `Building JIT module trtllm_utils` step with an idle
  CPU, while the head hit the cache and waited. Twice in a row. `AUTOTUNE=0`
  (`--no-enable-flashinfer-autotune`) gets past it. The FP8-dense numbers above use it on both sides.
- **Hardlinks and Docker bind mounts:** `os.link` fails across two separate `-v` mounts even on the
  same filesystem. Mount one common parent (`-v $HOME:/h`) when running `make-fp8-dense.py`.

## Credits

- **tonyd2wild** — SGLang lane, SM121 QSA guard fix, and the deploy-report conventions this repo follows.
- **foogitiff** — first dual-Spark vLLM bring-up (FP8 checkpoint) whose launcher this derives from.
- **RadixArk** — the NVFP4 checkpoint whose PLE layout turns out to match vLLM's own FP8-PLE method exactly.
- vLLM / Qwen teams for genuine day-0 multi-arch images.
- **orcarouter** — their [FP8-attention build](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4) of Flash-Next (and a PRO 6000 user's note about how fast it decodes) is what pointed us at the BF16 dense layers.
