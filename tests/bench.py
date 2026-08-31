from __future__ import annotations

import math
import statistics

import torch

from iris_kernel.torch_eager_kv import (
    apply_rope,
    quantize_int4_groupwise,
    pack_int4,
)

from iris_kernel.fused_kv import (
    _rope_fp16_paged_append,
    _quantize_pack_int4_g32,
    _rope_int4_paged_append,
)


@torch.no_grad()
def torch_eager_core(
    key,
    value,
    positions,
    slot_mapping,
    cos_cache,
    sin_cache,
    k_data,
    v_data,
    k_scale,
    v_scale,
    *,
    rotary_dim,
    group_size,
    page_size,
):
    slots = slot_mapping.long()
    pos = positions.long()

    pages = slots // page_size
    offsets = slots % page_size

    cos = cos_cache[pos][:, None, :]
    sin = sin_cache[pos][:, None, :]

    rotated_key = apply_rope(
        key,
        cos,
        sin,
        rotary_dim,
    )

    key_q, key_s = quantize_int4_groupwise(
        rotated_key,
        group_size,
    )

    value_q, value_s = quantize_int4_groupwise(
        value,
        group_size,
    )

    key_packed = pack_int4(key_q)
    value_packed = pack_int4(value_q)

    k_data[pages, offsets] = key_packed
    v_data[pages, offsets] = value_packed

    k_scale[pages, offsets] = key_s
    v_scale[pages, offsets] = value_s


def measure_cuda(
    fn,
    *,
    warmup: int = 20,
    samples: int = 50,
    inner: int = 100,
):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    times_us = []

    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(inner):
            fn()

        end.record()
        end.synchronize()

        elapsed_ms = start.elapsed_time(end)

        times_us.append(elapsed_ms * 1000.0 / inner)

    times_us.sort()

    p50 = statistics.median(times_us)
    p95 = times_us[int(0.95 * (len(times_us) - 1))]

    return p50, p95


def make_rope_cache(
    max_position: int,
    rotary_dim: int,
    device,
):
    position_ids = torch.arange(
        max_position,
        device=device,
        dtype=torch.float32,
    )

    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(
                0,
                rotary_dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            / rotary_dim
        )
    )

    freqs = position_ids[:, None] * inv_freq[None, :]

    return (
        freqs.cos().to(torch.float16),
        freqs.sin().to(torch.float16),
    )


def bench_case(
    num_tokens: int,
    *,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    group_size: int = 32,
    page_size: int = 16,
):
    device = torch.device("cuda")

    rotary_dim = head_dim

    num_pages = math.ceil(num_tokens / page_size)

    capacity = num_pages * page_size

    key = torch.randn(
        num_tokens,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

    value = torch.randn_like(key)

    positions = torch.arange(
        num_tokens,
        device=device,
        dtype=torch.int32,
    )

    slot_mapping = torch.arange(
        num_tokens,
        device=device,
        dtype=torch.int32,
    )

    cos_cache, sin_cache = make_rope_cache(
        max(num_tokens, 1),
        rotary_dim,
        device,
    )

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

    eager_k_data = torch.empty(
        data_shape,
        device=device,
        dtype=torch.uint8,
    )
    eager_v_data = torch.empty_like(eager_k_data)

    eager_k_scale = torch.empty(
        scale_shape,
        device=device,
        dtype=torch.float16,
    )
    eager_v_scale = torch.empty_like(eager_k_scale)

    def run_eager():
        torch_eager_core(
            key,
            value,
            positions,
            slot_mapping,
            cos_cache,
            sin_cache,
            eager_k_data,
            eager_v_data,
            eager_k_scale,
            eager_v_scale,
            rotary_dim=rotary_dim,
            group_size=group_size,
            page_size=page_size,
        )

    tmp_k = torch.empty(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

    tmp_v = torch.empty_like(tmp_k)

    unfused_k_data = torch.empty(
        data_shape,
        device=device,
        dtype=torch.uint8,
    )
    unfused_v_data = torch.empty_like(unfused_k_data)

    unfused_k_scale = torch.empty(
        scale_shape,
        device=device,
        dtype=torch.float16,
    )
    unfused_v_scale = torch.empty_like(unfused_k_scale)

    tmp_k_linear = tmp_k.view(
        capacity,
        num_kv_heads,
        head_dim,
    )[:num_tokens]

    tmp_v_linear = tmp_v.view(
        capacity,
        num_kv_heads,
        head_dim,
    )[:num_tokens]

    unfused_k_data_linear = unfused_k_data.view(
        capacity,
        num_kv_heads,
        head_dim // 2,
    )[:num_tokens]

    unfused_v_data_linear = unfused_v_data.view(
        capacity,
        num_kv_heads,
        head_dim // 2,
    )[:num_tokens]

    unfused_k_scale_linear = unfused_k_scale.view(
        capacity,
        num_kv_heads,
        head_dim // group_size,
    )[:num_tokens]

    unfused_v_scale_linear = unfused_v_scale.view(
        capacity,
        num_kv_heads,
        head_dim // group_size,
    )[:num_tokens]

    def run_unfused():
        _rope_fp16_paged_append(
            key,
            value,
            positions,
            slot_mapping,
            cos_cache,
            sin_cache,
            tmp_k,
            tmp_v,
            head_dim,
            rotary_dim,
            page_size,
            64,
        )

        _quantize_pack_int4_g32(
            tmp_k_linear,
            unfused_k_data_linear,
            unfused_k_scale_linear,
            head_dim,
            64,
        )

        _quantize_pack_int4_g32(
            tmp_v_linear,
            unfused_v_data_linear,
            unfused_v_scale_linear,
            head_dim,
            64,
        )

    fused_k_data = torch.empty(
        data_shape,
        device=device,
        dtype=torch.uint8,
    )
    fused_v_data = torch.empty_like(fused_k_data)

    fused_k_scale = torch.empty(
        scale_shape,
        device=device,
        dtype=torch.float16,
    )
    fused_v_scale = torch.empty_like(fused_k_scale)

    def run_fused():
        _rope_int4_paged_append(
            key,
            value,
            positions,
            slot_mapping,
            cos_cache,
            sin_cache,
            fused_k_data,
            fused_v_data,
            fused_k_scale,
            fused_v_scale,
            head_dim,
            page_size,
            64,
        )

    run_eager()
    run_unfused()
    run_fused()

    torch.cuda.synchronize()

    eager_k_data_valid = eager_k_data.view(
        capacity,
        num_kv_heads,
        head_dim // 2,
    )[:num_tokens]

    eager_v_data_valid = eager_v_data.view(
        capacity,
        num_kv_heads,
        head_dim // 2,
    )[:num_tokens]

    fused_k_data_valid = fused_k_data.view(
        capacity,
        num_kv_heads,
        head_dim // 2,
    )[:num_tokens]

    fused_v_data_valid = fused_v_data.view(
        capacity,
        num_kv_heads,
        head_dim // 2,
    )[:num_tokens]

    eager_k_scale_valid = eager_k_scale.view(
        capacity,
        num_kv_heads,
        head_dim // group_size,
    )[:num_tokens]

    eager_v_scale_valid = eager_v_scale.view(
        capacity,
        num_kv_heads,
        head_dim // group_size,
    )[:num_tokens]

    fused_k_scale_valid = fused_k_scale.view(
        capacity,
        num_kv_heads,
        head_dim // group_size,
    )[:num_tokens]

    fused_v_scale_valid = fused_v_scale.view(
        capacity,
        num_kv_heads,
        head_dim // group_size,
    )[:num_tokens]

    def report_packed(name, got, ref):
        diff = got != ref

        if torch.any(diff):
            idx = diff.nonzero()

            print(f"{name} mismatch count:", idx.shape[0])
            print(f"{name} first:", idx[:10].cpu())

            for row in idx[:5]:
                t, h, b = map(int, row.tolist())

                print(
                    f"{name} {(t, h, b)}:",
                    hex(int(got[t, h, b])),
                    hex(int(ref[t, h, b])),
                )

            return False

        return True

    k_ok = report_packed(
        "K",
        fused_k_data_valid,
        eager_k_data_valid,
    )

    v_ok = report_packed(
        "V",
        fused_v_data_valid,
        eager_v_data_valid,
    )

    assert k_ok
    assert v_ok

    if num_tokens <= 8:
        inner = 200
    elif num_tokens <= 32:
        inner = 100
    else:
        inner = 50

    eager_p50, eager_p95 = measure_cuda(
        run_eager,
        inner=inner,
    )

    unfused_p50, unfused_p95 = measure_cuda(
        run_unfused,
        inner=inner,
    )

    fused_p50, fused_p95 = measure_cuda(
        run_fused,
        inner=inner,
    )

    fp16_bytes, int4_bytes, saving = logical_kv_storage_bytes(
        head_dim,
        group_size,
    )

    return {
        "tokens": num_tokens,
        "eager_p50": eager_p50,
        "eager_p95": eager_p95,
        "unfused_p50": unfused_p50,
        "unfused_p95": unfused_p95,
        "fused_p50": fused_p50,
        "fused_p95": fused_p95,
        "vs_eager": eager_p50 / fused_p50,
        "vs_unfused": unfused_p50 / fused_p50,
        "tokens_per_sec": num_tokens / (fused_p50 * 1e-6),
        "saving": saving,
    }


def logical_kv_storage_bytes(
    head_dim: int,
    group_size: int,
) -> tuple[int, int, float]:
    fp16_size = torch.empty(
        (),
        dtype=torch.float16,
    ).element_size()

    uint8_size = torch.empty(
        (),
        dtype=torch.uint8,
    ).element_size()

    # K + V, each has head_dim FP16 values.
    fp16_bytes = 2 * head_dim * fp16_size

    # K + V, two INT4 values packed in one uint8.
    int4_data_bytes = 2 * (head_dim // 2) * uint8_size

    # One FP16 scale for every group_size values,
    # independently for K and V.
    scale_bytes = 2 * (head_dim // group_size) * fp16_size

    int4_bytes = int4_data_bytes + scale_bytes

    saving = (1.0 - int4_bytes / fp16_bytes) * 100.0

    return (
        fp16_bytes,
        int4_bytes,
        saving,
    )


def main():
    assert torch.cuda.is_available()

    prop = torch.cuda.get_device_properties(0)

    print(f"GPU: {prop.name}")
    print(
        "CUDA capability:",
        torch.cuda.get_device_capability(0),
    )
    print(
        "PyTorch:",
        torch.__version__,
    )
    print()

    print(
        "| T | eager p50 us | unfused p50 us | "
        "fused p50 us | vs eager | vs unfused | "
        "fused tok/s |"
    )

    print("|---:|---:|---:|---:|---:|---:|---:|")

    for tokens in [1, 8, 32, 128]:
        r = bench_case(tokens)

        print(
            f"| {r['tokens']} "
            f"| {r['eager_p50']:.3f} "
            f"| {r['unfused_p50']:.3f} "
            f"| {r['fused_p50']:.3f} "
            f"| {r['vs_eager']:.2f}x "
            f"| {r['vs_unfused']:.2f}x "
            f"| {r['tokens_per_sec']:,.0f} |"
            f"| {r['saving']:,.0f}"
        )

    print()


if __name__ == "__main__":
    main()
