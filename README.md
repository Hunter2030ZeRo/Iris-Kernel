# Iris-Kernel

**Compressed paged KV-cache kernels for NVIDIA GPUs, written in TileLang.**

Iris implements both sides of a compressed KV-cache pipeline:

```text
WRITE
FP16 K/V
  ↓
RoPE(K)
  ↓
group-wise INT4 quantization
  ↓
nibble packing
  ↓
paged KV cache

READ
paged INT4 K/V
  ↓
page lookup
  ↓
unpack + dequantize
  ↓
QK
  ↓
online softmax
  ↓
PV
  ↓
FP16 attention output
```

The central goal is to keep K/V compressed while they reside in global memory and consume them directly during decode attention, rather than reconstructing a full FP16 KV cache.

## Motivation

During autoregressive decoding, every new query attends to the accumulated K/V history.

As context length grows, repeatedly reading the historical KV cache becomes increasingly memory-intensive.

Simply compressing K/V while writing it is not sufficient if inference later performs:

```text
INT4 cache
    ↓
full FP16 dequantization
    ↓
temporary FP16 KV cache
    ↓
attention
```

That would reintroduce substantial global-memory traffic.

Iris therefore implements both:

- a **compressed KV producer**, and
- a **compressed KV consumer**.

The read-side kernel loads packed INT4 K/V and dequantizes them directly into the attention computation.

## INT4 cache format

The current format uses symmetric signed INT4 values:

```text
range: [-7, 7]
packing: two INT4 values per uint8
order: low nibble first
group size: 32
scale type: FP16
```

Physical cache layout:

```text
k_data / v_data:
[num_pages, page_size, Hkv, D / 2] uint8

k_scale / v_scale:
[num_pages, page_size, Hkv, D / 32] fp16
```

## Logical KV storage

For one token and one KV head with head dimension `D`:

### FP16 K/V

K and V each contain `D` FP16 values:

```text
2 × D × 2 bytes = 4D bytes
```

### Iris INT4 K/V

Packed K/V:

```text
2 × D × 4 bits = D bytes
```

FP16 group scales with group size `G`:

```text
2 × (D / G) × 2 bytes = 4D/G bytes
```

Total:

```text
D + 4D/G bytes
```

Therefore the logical storage reduction is:

```text
1 - (D + 4D/G) / 4D
= 3/4 - 1/G
```

For the current `G = 32` format:

```text
71.875% logical KV storage reduction
```

For example, at `D = 128`:

```text
FP16 K/V:       512 bytes
Iris INT4 K/V:  144 bytes
```

This is a **logical populated-token storage figure**. Actual allocated memory may additionally contain unused capacity in partially filled pages.

## Write-side fused kernel

The write path performs:

```text
FP16 K
  → RoPE
  → group-wise symmetric INT4 quantization
  → nibble packing
  → paged K cache

FP16 V
  → group-wise symmetric INT4 quantization
  → nibble packing
  → paged V cache
```

The fused path avoids materializing intermediate FP16 transformed K/V tensors in global memory.

Current quantization contract:

1. compute group absolute maximum in FP32,
2. calculate the scale,
3. round the stored scale to FP16,
4. use that stored FP16 scale for quantization,
5. round to nearest with ties-to-even,
6. clamp to `[-7, 7]`,
7. pack two values into each `uint8`.

Using the stored FP16 scale for both encoding and decoding keeps the format contract deterministic.

## Read-side fused decode attention

Iris also implements a decode-only attention consumer for the compressed cache.

```text
FP16 Query
+
Paged INT4 K/V
+
FP16 scales
       ↓
physical page lookup
       ↓
nibble unpack
       ↓
signed INT4 decode
       ↓
group dequantization
       ↓
QK
       ↓
online softmax
       ↓
PV accumulation
       ↓
FP16 output
```

Critically, the complete FP16 K/V cache is **not written back to global memory**.

Packed K/V is decoded into local/shared tiles and immediately consumed by attention.

## Online softmax

The attention kernel processes the KV sequence in tiles while maintaining:

- a running score maximum,
- a running normalization sum,
- and a rescaled output accumulator.

This avoids storing the full attention-score matrix in global memory.

Conceptually:

```text
for each KV tile:
    scores = Q @ Kᵀ

    update running maximum
    rescale previous normalization/output

    probabilities = exp(scores - running_max)

    update normalization sum
    output += probabilities @ V

output /= normalization_sum
```

## Hardware-aware schedule

The initial decode fast path uses:

```text
CTA mapping:
    one (sequence, KV head) per CTA

query tile:
    up to 16 query heads sharing one KV head

KV tile:
    32 tokens

physical page:
    16 tokens

accumulation:
    FP32
```

A 32-token attention tile therefore spans two physical KV pages.

QK and PV are expressed with TileLang GEMM primitives, while unpack/dequantization is fused into cache loading.

## Supported decode scope

Current fast path:

- decode only: one query token per sequence,
- GQA / MQA / MHA semantics,
- `Hq % Hkv == 0`,
- page size `16`,
- INT4 group size `32`,
- head dimension `64` or `128`,
- up to 16 query heads per KV head,
- CUDA compute capability 7.0 or newer.

The current implementation targets conventional NVIDIA CUDA GPUs rather than a single GPU generation.

## Correctness

The repository contains independent PyTorch eager references for:

- INT4 unpacking,
- group-wise dequantization,
- paged compressed attention,
- and FP16 paged attention.

The tests cover:

- compressed write correctness,
- packed K/V equality,
- stored scale correctness,
- non-contiguous physical page tables,
- variable sequence lengths,
- GQA head mapping,
- fused INT4 attention vs. the eager INT4 reference,
- INT4 attention output quality vs. FP16 attention.

Run:

```bash
uv run pytest -q
```

## Benchmarks

### Write path

`tests/bench.py` compares:

```text
PyTorch eager reference

vs.

TileLang unfused:
    RoPE/write
    + K quantization
    + V quantization

vs.

TileLang fused:
    RoPE
    + K/V quantization
    + packing
    + paged write
```

Run:

```bash
uv run python tests/bench.py
```

### Read path

`tests/bench_attention.py` compares:

```text
PyTorch eager INT4 dequantization + attention

vs.

matched fused FP16 paged decode attention

vs.

Iris fused INT4 paged decode attention
```

Default configuration:

```text
B    = 1
Hq   = 32
Hkv  = 8
D    = 128
page = 16
group = 32

sequence lengths:
128
512
2048
4096
```

Run:

```bash
uv run python tests/bench_attention.py
```

The benchmark reports:

- median latency,
- INT4 vs. eager speed ratio,
- INT4 vs. matched FP16 speed ratio,
- output cosine similarity,
- output RMSE.

The most important comparison is:

```text
fused INT4 paged attention
vs.
matched fused FP16 paged attention
```

because both use the same attention structure and differ primarily in KV representation and decode cost.

<!--
Paste the final measured tests/bench_attention.py table here.
Do not manually invent or normalize benchmark values.
-->

## Project structure

```text
src/iris_kernel/
├── fused_kv.py
│   ├── FP16 RoPE paged append
│   ├── INT4 quantize/pack
│   └── fused RoPE + INT4 paged append
│
├── fused_attention.py
│   ├── fused INT4 paged decode attention
│   └── matched FP16 paged decode attention
│
├── torch_eager_kv.py
│   └── write-side reference
│
└── torch_eager_attention.py
    └── attention reference

tests/
├── test_correctness.py
├── test_attention.py
├── bench.py
└── bench_attention.py
```

## Scope and limitations

Iris is a kernel prototype, not a complete inference engine.

The current implementation does **not** claim:

- end-to-end model decode acceleration,
- prefill attention optimization,
- training/backward support,
- page allocation or eviction,
- split-KV scheduling,
- integration with a production inference runtime,
- or a new attention algorithm.

The project instead isolates one systems problem:

> Can a paged KV cache remain compressed in global memory and be consumed directly by a fused attention kernel without reconstructing the full FP16 cache?

Iris implements both the producer and consumer paths needed to evaluate that question.

## Installation

Python 3.12 or newer is required.

```bash
git clone https://github.com/Hunter2030ZeRo/Iris-Kernel
cd Iris-Kernel
uv sync
```

```bash
uv run pytest -q
uv run python tests/bench.py
uv run python tests/bench_attention.py
```

## Implementation

- Python
- PyTorch
- TileLang(CUDA)
