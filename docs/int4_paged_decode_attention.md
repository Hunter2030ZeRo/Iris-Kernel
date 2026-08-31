# INT4 Paged Decode Attention

## Why this exists

The original Iris writer fuses RoPE, group-wise INT4 quantization, nibble packing, and paged KV-cache append. That reduces logical KV storage and removes write-side intermediates, but it does not by itself reduce the dominant read traffic of long-context decoding.

This extension adds the missing consumer path: a decode-only grouped-query attention kernel that reads Iris's compressed paged K/V cache directly and performs the complete attention calculation without materializing a dequantized FP16 cache in global memory.

## Fused pipeline

```text
FP16 query
+ paged packed INT4 K/V
+ FP16 group scales
        |
        v
page lookup -> nibble unpack -> signed INT4 decode -> group dequantization
        -> QK -> online softmax -> PV -> FP16 output
```

The kernel therefore fuses the parts that matter for cache consumption: dequantization, score computation, normalization, and value accumulation.

## Scope

- Decode only: one query token per sequence (`query: [B, Hq, D]`).
- GQA/MQA/MHA semantics: `Hq` must be divisible by `Hkv`.
- Paged NHD cache layout.
- Page size 16.
- Symmetric signed INT4 in `[-7, 7]`, low nibble first.
- One FP16 scale per 32 values.
- Initial fast paths for head dimensions 64 and 128.
- CUDA compute capability 7.0 or newer; primary validation targets are Turing T4 and Ada RTX 4070 Laptop.

## Hardware-aware schedule

- One CTA handles one `(sequence, KV head)` pair.
- Up to 16 query heads sharing the KV head are processed as one GQA tile.
- A 32-token attention tile spans two physical 16-token pages.
- Packed K/V is unpacked and dequantized directly into shared-memory tiles.
- QK and PV use TileLang GEMM primitives with FP32 accumulation.
- Online softmax maintains running maximum, normalization sum, and rescaled output accumulator, so no score matrix is written to global memory.
- The INT4 path and matched FP16 path use the same attention tile and online-softmax schedule, allowing a direct read-bandwidth/dequantization trade-off comparison.

## Validation

`tests/test_attention.py` checks:

1. End-to-end fused INT4 attention against a PyTorch eager reference that reads the same compressed cache.
2. GQA head mapping.
3. Non-contiguous physical page tables and variable sequence lengths.
4. INT4 output quality against an FP16 paged-cache reference.

## Benchmark

`tests/bench_attention.py` compares:

- PyTorch eager INT4 dequantization + attention.
- Matched fused FP16 paged decode attention.
- Fused INT4 paged decode attention.

The default benchmark uses `B=1`, `Hq=32`, `Hkv=8`, `D=128`, page size 16, and sequence lengths 128, 512, 2048, and 4096. It reports median latency, INT4-vs-FP16 speed ratio, output cosine similarity, and RMSE.

## Deliberate non-claims

This is not a new attention algorithm and does not claim full-model decode speedup until integration with a real inference engine is measured. It is a compressed paged-cache producer/consumer kernel set. Prefill attention, backward propagation, split-KV scheduling, and end-to-end model integration remain outside the current scope.
