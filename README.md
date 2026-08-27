# Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark
First vLLM deployment of Qwen3.8-Flash-Next (RadixArk NVFP4) on 2× DGX Spark (GB10/SM121) — TP2+EP, MTP speculative decode, CUDA graphs, 262K context. Includes the 3-line PLE FP8 quant-resolver patch the official day-0 image needs, launcher, benchmarks, and gotchas for multi-pair Spark fleets.
