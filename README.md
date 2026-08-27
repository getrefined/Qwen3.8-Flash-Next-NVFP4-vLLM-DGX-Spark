# Qwen3.8-Flash-Next NVFP4 on 2× DGX Spark — vLLM (TP2+EP, MTP3, CUDA graphs)

First working **vLLM** deployment of `RadixArk/Qwen3.8-Flash-Next-NVFP4` on **2× NVIDIA DGX Spark (GB10 / SM121)**, using the official day-0 image `vllm/vllm-openai:qwen38-flash-next` plus **one three-line patch** to the PLE quant-method resolver. Brought up 2026-08-27, the day after the model dropped.

This is the vLLM sibling of tonyd2wild's SGLang deployment
([Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)) —
same checkpoint, same hardware class, different engine. The launcher derives from
foogitiff's dual-Spark FP8 config (NVIDIA forum, Qwen3.8-Flash-Next thread, post 97),
adapted for the NVFP4 checkpoint.

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

## Credits

- **tonyd2wild** — SGLang lane, SM121 QSA guard fix, and the deploy-report conventions this repo follows.
- **foogitiff** — first dual-Spark vLLM bring-up (FP8 checkpoint) whose launcher this derives from.
- **RadixArk** — the NVFP4 checkpoint whose PLE layout turns out to match vLLM's own FP8-PLE method exactly.
- vLLM / Qwen teams for genuine day-0 multi-arch images.
