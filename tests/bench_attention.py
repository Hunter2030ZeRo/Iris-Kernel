from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import torch

from iris_kernel.fused_attention import (
    fused_fp16_paged_decode_attention,
    fused_int4_paged_decode_attention,
)
from iris_kernel.torch_eager_attention import (
    torch_eager_fp16_paged_decode_attention,
    torch_eager_int4_paged_decode_attention,
)
from iris_kernel.torch_eager_kv import apply_rope, torch_eager_append


def measure_cuda(fn, *, warmup: int = 20, samples: int = 50, inner: int = 20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_us = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0 / inner)

    samples_us.sort()
    return {
        "p50_us": statistics.median(samples_us),
        "p95_us": samples_us[int(0.95 * (len(samples_us) - 1))],
    }


def make_case(
    seq_len: int,
    *,
    batch: int = 1,
    num_query_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    group_size: int = 32,
):
    torch.manual_seed(3407 + seq_len)
    device = torch.device("cuda")
    logical_pages = math.ceil(seq_len / page_size)
    num_pages = batch * logical_pages

    page_ids = torch.randperm(num_pages, device=device, dtype=torch.int64).reshape(
        batch, logical_pages
    )
    block_table = page_ids.to(torch.int32)
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)

    positions_all = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
            / head_dim
        )
    )
    freqs = positions_all[:, None] * inv_freq[None, :]
    cos_cache = freqs.cos().to(torch.float16)
    sin_cache = freqs.sin().to(torch.float16)

    data_shape = (
        num_pages,
        page_size,
        num_kv_heads,
        head_dim // 2,
    )
    scale_shape = (
        num_pages,
        page_size,
        num_kv_heads,
        head_dim // group_size,
    )
    k_data = torch.empty(data_shape, device=device, dtype=torch.uint8)
    v_data = torch.empty_like(k_data)
    k_scale = torch.empty(scale_shape, device=device, dtype=torch.float16)
    v_scale = torch.empty_like(k_scale)
    fp16_k_cache = torch.empty(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    fp16_v_cache = torch.empty_like(fp16_k_cache)

    flat_key = []
    flat_value = []
    flat_positions = []
    flat_slots = []

    for batch_idx in range(batch):
        key = torch.randn(
            seq_len,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.float16,
        )
        value = torch.randn_like(key)
        positions = torch.arange(seq_len, device=device, dtype=torch.int32)

        logical_page = torch.div(positions, page_size, rounding_mode="floor")
        offset = positions % page_size
        physical_page = block_table[batch_idx, logical_page.long()]
        slots = physical_page * page_size + offset

        rotated_key = apply_rope(
            key,
            cos_cache[:, None, :],
            sin_cache[:, None, :],
            rotary_dim=head_dim,
        )
        fp16_k_cache[physical_page.long(), offset.long()] = rotated_key
        fp16_v_cache[physical_page.long(), offset.long()] = value

        flat_key.append(key)
        flat_value.append(value)
        flat_positions.append(positions)
        flat_slots.append(slots.to(torch.int32))

    torch_eager_append(
        torch.cat(flat_key, dim=0),
        torch.cat(flat_value, dim=0),
        torch.cat(flat_positions, dim=0),
        torch.cat(flat_slots, dim=0),
        cos_cache,
        sin_cache,
        k_data,
        v_data,
        k_scale,
        v_scale,
        rotary_dim=head_dim,
        group_size=group_size,
        page_size=page_size,
    )

    query_unrotated = torch.randn(
        batch,
        num_query_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    query = apply_rope(
        query_unrotated,
        cos_cache[seq_lens.long() - 1][:, None, :],
        sin_cache[seq_lens.long() - 1][:, None, :],
        rotary_dim=head_dim,
    )

    return {
        "query": query,
        "k_data": k_data,
        "v_data": v_data,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "fp16_k_cache": fp16_k_cache,
        "fp16_v_cache": fp16_v_cache,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "page_size": page_size,
        "group_size": group_size,
    }


def bench_case(seq_len: int):
    case = make_case(seq_len)
    output_int4 = torch.empty_like(case["query"])
    output_fp16 = torch.empty_like(case["query"])

    def run_eager_int4():
        torch_eager_int4_paged_decode_attention(
            case["query"],
            case["k_data"],
            case["v_data"],
            case["k_scale"],
            case["v_scale"],
            case["block_table"],
            case["seq_lens"],
            page_size=case["page_size"],
            group_size=case["group_size"],
        )

    def run_fused_int4():
        fused_int4_paged_decode_attention(
            case["query"],
            case["k_data"],
            case["v_data"],
            case["k_scale"],
            case["v_scale"],
            case["block_table"],
            case["seq_lens"],
            output_int4,
            page_size=case["page_size"],
            group_size=case["group_size"],
        )

    def run_fused_fp16():
        fused_fp16_paged_decode_attention(
            case["query"],
            case["fp16_k_cache"],
            case["fp16_v_cache"],
            case["block_table"],
            case["seq_lens"],
            output_fp16,
            page_size=case["page_size"],
        )

    eager_expected = torch_eager_int4_paged_decode_attention(
        case["query"],
        case["k_data"],
        case["v_data"],
        case["k_scale"],
        case["v_scale"],
        case["block_table"],
        case["seq_lens"],
        page_size=case["page_size"],
        group_size=case["group_size"],
    )
    fp16_expected = torch_eager_fp16_paged_decode_attention(
        case["query"],
        case["fp16_k_cache"],
        case["fp16_v_cache"],
        case["block_table"],
        case["seq_lens"],
        page_size=case["page_size"],
    )

    run_fused_int4()
    run_fused_fp16()
    torch.cuda.synchronize()
    torch.testing.assert_close(output_int4, eager_expected, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(output_fp16, fp16_expected, rtol=5e-2, atol=5e-2)

    cosine = torch.nn.functional.cosine_similarity(
        output_int4.float().flatten(0, 1),
        output_fp16.float().flatten(0, 1),
        dim=-1,
    ).mean()
    rmse = torch.mean((output_int4.float() - output_fp16.float()) ** 2).sqrt()

    inner = 50 if seq_len <= 512 else 20
    eager = measure_cuda(run_eager_int4, inner=max(1, inner // 5))
    fp16 = measure_cuda(run_fused_fp16, inner=inner)
    int4 = measure_cuda(run_fused_int4, inner=inner)

    return {
        "seq_len": seq_len,
        "eager_int4_p50_us": eager["p50_us"],
        "fused_fp16_p50_us": fp16["p50_us"],
        "fused_int4_p50_us": int4["p50_us"],
        "vs_eager": eager["p50_us"] / int4["p50_us"],
        "vs_fp16": fp16["p50_us"] / int4["p50_us"],
        "output_cosine": float(cosine),
        "output_rmse": float(rmse),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()
    print(
        "| L | eager INT4 us | fused FP16 us | fused INT4 us | "
        "vs eager | vs FP16 | cosine | RMSE |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")

    results = []
    for seq_len in (128, 512, 2048, 4096):
        result = bench_case(seq_len)
        results.append(result)
        print(
            f"| {seq_len} | {result['eager_int4_p50_us']:.3f} "
            f"| {result['fused_fp16_p50_us']:.3f} "
            f"| {result['fused_int4_p50_us']:.3f} "
            f"| {result['vs_eager']:.2f}x "
            f"| {result['vs_fp16']:.2f}x "
            f"| {result['output_cosine']:.6f} "
            f"| {result['output_rmse']:.6f} |"
        )

    output_path = Path("attention_benchmark.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
