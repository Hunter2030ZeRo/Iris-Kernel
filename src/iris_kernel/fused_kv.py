from __future__ import annotations

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda")
def _rope_fp16_paged_append(
    key,
    value,
    positions,
    slot_mapping,
    cos_cache,
    sin_cache,
    k_cache,
    v_cache,
    head_dim: int,
    rotary_dim: int,
    page_size: int,
    threads: int = 64,
):
    NT, H, MP, NP = T.const("NT, H, MP, NP")

    key: T.Tensor(
        (NT, H, head_dim),
        T.float16,
    )  # ty:ignore[invalid-type-form]
    value: T.Tensor(
        (NT, H, head_dim),
        T.float16,
    )  # ty:ignore[invalid-type-form]
    positions: T.Tensor(
        (NT,),
        T.int32,
    )  # ty:ignore[invalid-type-form]

    slot_mapping: T.Tensor(
        (NT,),
        T.int32,
    )  # ty:ignore[invalid-type-form]

    cos_cache: T.Tensor(
        (MP, rotary_dim // 2),
        T.float16,
    )  # ty:ignore[invalid-type-form]
    sin_cache: T.Tensor(
        (MP, rotary_dim // 2),
        T.float16,
    )  # ty:ignore[invalid-type-form]
    k_cache: T.Tensor(
        (NP, page_size, H, head_dim),
        T.float16,
    )  # ty:ignore[invalid-type-form]
    v_cache: T.Tensor(
        (NP, page_size, H, head_dim),
        T.float16,
    )  # ty:ignore[invalid-type-form]

    half = rotary_dim // 2

    with T.Kernel(NT, H, threads=threads) as (token_idx, head_idx):
        slot = slot_mapping[token_idx]

        if slot >= 0:
            page = slot // page_size
            offset = slot % page_size

            position = positions[token_idx]

            for i in T.Parallel(half):
                x1 = key[
                    token_idx,
                    head_idx,
                    i,
                ]

                x2 = key[
                    token_idx,
                    head_idx,
                    i + half,
                ]

                cos = cos_cache[position, i]
                sin = sin_cache[position, i]

                y1 = x1 * cos - x2 * sin
                y2 = x2 * cos + x1 * sin

                k_cache[
                    page,
                    offset,
                    head_idx,
                    i,
                ] = y1

                k_cache[
                    page,
                    offset,
                    head_idx,
                    i + half,
                ] = y2

                v_cache[
                    page,
                    offset,
                    head_idx,
                    i,
                ] = value[
                    token_idx,
                    head_idx,
                    i,
                ]

                v_cache[
                    page,
                    offset,
                    head_idx,
                    i + half,
                ] = value[
                    token_idx,
                    head_idx,
                    i + half,
                ]

            if rotary_dim < head_dim:
                for i in T.Parallel(head_dim - rotary_dim):
                    dim = rotary_dim + i

                    k_cache[
                        page,
                        offset,
                        head_idx,
                        dim,
                    ] = key[
                        token_idx,
                        head_idx,
                        dim,
                    ]

                    v_cache[
                        page,
                        offset,
                        head_idx,
                        dim,
                    ] = value[
                        token_idx,
                        head_idx,
                        dim,
                    ]


def fused_rope_fp16_append(
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    rotary_dim: int,
    page_size: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Iris fused kernels require CUDA")

    if key.device.type != "cuda":
        raise ValueError("key/value must be CUDA tensors")

    if key.shape != value.shape:
        raise ValueError("key and value must have identical shapes")

    if key.ndim != 3:
        raise ValueError("key/value must have shape [T, Hkv, D]")

    if key.dtype != torch.float16:
        raise TypeError("Stage A currently supports FP16 input")

    if value.dtype != torch.float16:
        raise TypeError("Stage A currently supports FP16 input")

    num_tokens, num_kv_heads, head_dim = key.shape

    if rotary_dim > head_dim:
        raise ValueError("rotary_dim cannot exceed head_dim")

    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")

    if page_size <= 0:
        raise ValueError("page_size must be positive")

    if positions.shape != (num_tokens,):
        raise ValueError("positions must have shape [T]")

    if slot_mapping.shape != (num_tokens,):
        raise ValueError("slot_mapping must have shape [T]")

    expected_cache_tail = (
        page_size,
        num_kv_heads,
        head_dim,
    )

    if tuple(k_cache.shape[1:]) != expected_cache_tail:
        raise ValueError("invalid k_cache shape")

    if tuple(v_cache.shape) != tuple(k_cache.shape):
        raise ValueError("k_cache and v_cache must have identical shapes")

    if k_cache.dtype != torch.float16:
        raise TypeError("Stage A cache must use FP16")

    if v_cache.dtype != torch.float16:
        raise TypeError("Stage A cache must use FP16")

    device = key.device

    for name, tensor in (
        ("value", value),
        ("positions", positions),
        ("slot_mapping", slot_mapping),
        ("cos_cache", cos_cache),
        ("sin_cache", sin_cache),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")

    # Stage A standardizes metadata as int32 because the
    # final kernel will use these directly in device code.
    if positions.dtype != torch.int32:
        raise TypeError("positions must use torch.int32")

    if slot_mapping.dtype != torch.int32:
        raise TypeError("slot_mapping must use torch.int32")

    half = rotary_dim // 2

    if cos_cache.ndim != 2:
        raise ValueError("cos_cache must have shape [max_position, rotary_dim // 2]")

    if sin_cache.shape != cos_cache.shape:
        raise ValueError("sin_cache must match cos_cache")

    if cos_cache.shape[-1] != half:
        raise ValueError(f"RoPE cache width must be {half}")

    if cos_cache.dtype != torch.float16:
        raise TypeError("Stage A RoPE cache must use FP16")

    if sin_cache.dtype != torch.float16:
        raise TypeError("Stage A RoPE cache must use FP16")

    capability = torch.cuda.get_device_capability(key.device)

    if capability < (7, 0):
        raise RuntimeError(f"unsupported CUDA compute capability: {capability}")

    # The kernel itself handles slot < 0.
    #
    # Other metadata validation stays on the host.
    valid_slots = slot_mapping[slot_mapping >= 0]
    valid_positions = positions[slot_mapping >= 0]

    if valid_slots.numel() > 0:
        if torch.unique(valid_slots).numel() != valid_slots.numel():
            raise ValueError("duplicate physical KV-cache slots are not allowed")

        max_slot = k_cache.shape[0] * page_size

        if bool(torch.any(valid_slots >= max_slot)):
            raise IndexError("slot_mapping points outside allocated cache")

    if bool(torch.any(valid_positions < 0)):
        raise IndexError("positions must be non-negative")

    if bool(torch.any(valid_positions >= cos_cache.shape[0])):
        raise IndexError("positions point outside RoPE cache")

    _rope_fp16_paged_append(
        key,
        value,
        positions,
        slot_mapping,
        cos_cache,
        sin_cache,
        k_cache,
        v_cache,
        head_dim,
        rotary_dim,
        page_size,
        64,
    )


@tilelang.jit(target="cuda")
def _quantize_pack_int4_g32(
    x,
    packed,
    scales,
    head_dim: int,
    threads: int = 64,
):
    NT, H = T.const("NT, H")

    x: T.Tensor((NT, H, head_dim), T.float16)  # ty:ignore[invalid-type-form]
    packed: T.Tensor((NT, H, head_dim // 2), T.uint8)  # ty:ignore[invalid-type-form]
    scales: T.Tensor((NT, H, head_dim // 32), T.float16)  # ty:ignore[invalid-type-form]

    with T.Kernel(NT, H, threads=64) as (token_idx, head_idx):
        lane = T.get_lane_idx()
        warp = T.get_warp_idx()

        for chunk in T.serial(head_dim // 64):
            group = chunk * 2 + warp
            dim = group * 32 + lane

            value = T.cast(
                x[token_idx, head_idx, dim],
                T.float32,
            )

            abs_value = T.abs(value)
            amax = T.warp_reduce_max(abs_value)

            scale32 = T.max(amax / 7.0, 6.103515625e-05)

            scale16 = T.cast(scale32, T.float16)

            quant_scale = T.cast(scale16, T.float32)

            if lane == 0:
                scales[
                    token_idx,
                    head_idx,
                    group,
                ] = scale16

            q_rounded = T.round(value / quant_scale, rounding_mode="ties-to-even")
            q_clamped = T.max(-7.0, T.min(7.0, q_rounded))

            q = T.cast(q_clamped, T.int32)

            q_hi = T.shfl_down(q, 1)

            if lane % 2 == 0:
                lo = q & 0xF  # ty:ignore[unsupported-operator]
                hi = q_hi & 0xF

                byte = lo | (hi << 4)

                byte_dim = group * 16 + lane // 2

                packed[
                    token_idx,
                    head_idx,
                    byte_dim,
                ] = T.cast(byte, T.uint8)


def fused_quantize_pack_int4(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Iris fused kernels require CUDA")

    if x.device.type != "cuda":
        raise ValueError("x must be a CUDA tensor")

    if x.ndim != 3:
        raise ValueError("x must have shape [T, H_kv, D]")

    if x.dtype != torch.float16:
        raise TypeError("INT4 fast path currently requires FP16 input")

    _, _, head_dim = x.shape

    if head_dim % 64 != 0:
        raise NotImplementedError(
            "INT4 fast path currently requires head_dim divisible by 64"
        )

    expected_packed = (x.shape[0], x.shape[1], head_dim // 2)
    expected_scales = (x.shape[0], x.shape[1], head_dim // 32)

    if tuple(packed.shape) != expected_packed:
        raise ValueError(f"packed tensor must have shape {expected_packed}")

    if tuple(scales.shape) != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}")

    if packed.dtype != torch.uint8:
        raise TypeError("packed tensor must use UINT8(torch.uint8)")

    if scales.dtype != torch.float16:
        raise TypeError("scales must use FP16(torch.float16)")

    if packed.device != x.device:
        raise ValueError("packed tensor must be on the same device as x")

    if scales.device != x.device:
        raise ValueError("scales must be on the same device as x")

    _quantize_pack_int4_g32(
        x,
        packed,
        scales,
        head_dim,
        64,
    )


@tilelang.jit(target="cuda")
def _rope_int4_paged_append(
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
    head_dim: int,
    page_size: int,
    threads: int = 64,
):
    NT, H, MP, NP = T.const("NT, H, MP, NP")

    key: T.Tensor((NT, H, head_dim), T.float16)  # ty:ignore[invalid-type-form]
    value: T.Tensor((NT, H, head_dim), T.float16)  # ty:ignore[invalid-type-form]

    positions: T.Tensor((NT,), T.int32)  # ty:ignore[invalid-type-form]
    slot_mapping: T.Tensor((NT,), T.int32)  # ty:ignore[invalid-type-form]

    cos_cache: T.Tensor((MP, head_dim // 2), T.float16)  # ty:ignore[invalid-type-form]
    sin_cache: T.Tensor((MP, head_dim // 2), T.float16)  # ty:ignore[invalid-type-form]

    k_data: T.Tensor((NP, page_size, H, head_dim // 2), T.uint8)  # ty:ignore[invalid-type-form]
    v_data: T.Tensor((NP, page_size, H, head_dim // 2), T.uint8)  # ty:ignore[invalid-type-form]

    k_scale: T.Tensor((NP, page_size, H, head_dim // 32), T.float16)  # ty:ignore[invalid-type-form]
    v_scale: T.Tensor((NP, page_size, H, head_dim // 32), T.float16)  # ty:ignore[invalid-type-form]

    half = head_dim // 2

    with T.Kernel(NT, H, threads=threads) as (token_idx, head_idx):
        slot = slot_mapping[token_idx]

        if slot >= 0:
            page = slot // page_size
            offset = slot % page_size
            position = positions[token_idx]

            lane = T.get_lane_idx()
            warp = T.get_warp_idx()

            for chunk in T.unroll(head_dim // 64, explicit=True):
                group = chunk * 2 + warp
                dim = group * 32 + lane

                pair_idx = dim % half

                x1 = key[
                    token_idx,
                    head_idx,
                    pair_idx,
                ]

                x2 = key[
                    token_idx,
                    head_idx,
                    pair_idx + half,
                ]

                cos = cos_cache[
                    position,
                    pair_idx,
                ]

                sin = sin_cache[
                    position,
                    pair_idx,
                ]

                x1_cos = T.cast(
                    T.cast(x1, T.float32) * T.cast(cos, T.float32),
                    T.float16,
                )

                x2_sin = T.cast(
                    T.cast(x2, T.float32) * T.cast(sin, T.float32),
                    T.float16,
                )

                x2_cos = T.cast(
                    T.cast(x2, T.float32) * T.cast(cos, T.float32),
                    T.float16,
                )

                x1_sin = T.cast(
                    T.cast(x1, T.float32) * T.cast(sin, T.float32),
                    T.float16,
                )

                k_rot_lo = T.cast(
                    T.cast(x1_cos, T.float32) - T.cast(x2_sin, T.float32),
                    T.float16,
                )

                k_rot_hi = T.cast(
                    T.cast(x2_cos, T.float32) + T.cast(x1_sin, T.float32),
                    T.float16,
                )

                k16 = T.if_then_else(
                    dim < half,
                    k_rot_lo,
                    k_rot_hi,
                )

                k_value = T.cast(
                    k16,
                    T.float32,
                )
                v_value = T.cast(value[token_idx, head_idx, dim], T.float32)

                k_amax = T.warp_reduce_max(T.abs(k_value))
                v_amax = T.warp_reduce_max(T.abs(v_value))

                k_scale32 = T.max(k_amax / 7.0, 6.103515625e-05)
                v_scale32 = T.max(v_amax / 7.0, 6.103515625e-05)

                k_scale16 = T.cast(k_scale32, T.float16)
                v_scale16 = T.cast(v_scale32, T.float16)

                k_quant_scale = T.cast(k_scale16, T.float32)
                v_quant_scale = T.cast(v_scale16, T.float32)

                if lane == 0:
                    k_scale[
                        page,
                        offset,
                        head_idx,
                        group,
                    ] = k_scale16

                    v_scale[
                        page,
                        offset,
                        head_idx,
                        group,
                    ] = v_scale16

                k_rounded = T.round(
                    k_value / k_quant_scale, rounding_mode="ties-to-even"
                )
                v_rounded = T.round(
                    v_value / v_quant_scale, rounding_mode="ties-to-even"
                )

                k_clamped = T.max(-7.0, T.min(7.0, k_rounded))
                v_clamped = T.max(-7.0, T.min(7.0, v_rounded))

                k_q = T.cast(k_clamped, T.int32)
                v_q = T.cast(v_clamped, T.int32)

                k_q_hi = T.shfl_down(k_q, 1)
                v_q_hi = T.shfl_down(v_q, 1)

                if lane % 2 == 0:
                    byte_idx = group * 16 + lane // 2

                    k_byte = (k_q & 0xF) | ((k_q_hi & 0xF) << 4)  # ty:ignore[unsupported-operator]

                    v_byte = (v_q & 0xF) | ((v_q_hi & 0xF) << 4)  # ty:ignore[unsupported-operator]

                    k_data[
                        page,
                        offset,
                        head_idx,
                        byte_idx,
                    ] = T.cast(k_byte, T.uint8)

                    v_data[
                        page,
                        offset,
                        head_idx,
                        byte_idx,
                    ] = T.cast(v_byte, T.uint8)


def fused_rope_int4_paged_append(
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    rotary_dim: int,
    group_size: int = 32,
    page_size: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Iris fused kernels require CUDA")

    if key.device.type != "cuda":
        raise ValueError("key/value must be CUDA tensors")

    if key.shape != value.shape:
        raise ValueError("key and value must have identical shapes")

    if key.ndim != 3:
        raise ValueError("key/value must have shape [T, Hkv, D]")

    if key.dtype != torch.float16:
        raise TypeError("key must use FP16(torch.float16)")

    if value.dtype != torch.float16:
        raise TypeError("value must use FP16(torch.float16)")

    num_tokens, num_kv_heads, head_dim = key.shape

    # Current Stage C fast path.
    if head_dim % 64 != 0:
        raise NotImplementedError("Stage C currently requires head_dim divisible by 64")

    if group_size != 32:
        raise NotImplementedError("Stage C currently supports group_size=32")

    # For now, keep Stage C to full RoPE.
    if rotary_dim != head_dim:
        raise NotImplementedError("Stage C currently requires rotary_dim == head_dim")

    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")

    if page_size <= 0:
        raise ValueError("page_size must be positive")

    if positions.shape != (num_tokens,):
        raise ValueError("positions must have shape [T]")

    if slot_mapping.shape != (num_tokens,):
        raise ValueError("slot_mapping must have shape [T]")

    device = key.device

    for name, tensor in (
        ("value", value),
        ("positions", positions),
        ("slot_mapping", slot_mapping),
        ("cos_cache", cos_cache),
        ("sin_cache", sin_cache),
        ("k_data", k_data),
        ("v_data", v_data),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")

    if positions.dtype != torch.int32:
        raise TypeError("positions must use torch.int32")

    if slot_mapping.dtype != torch.int32:
        raise TypeError("slot_mapping must use torch.int32")

    half = rotary_dim // 2

    if cos_cache.ndim != 2:
        raise ValueError("cos_cache must have shape [max_position, rotary_dim // 2]")

    if sin_cache.shape != cos_cache.shape:
        raise ValueError("sin_cache must match cos_cache")

    if cos_cache.shape[-1] != half:
        raise ValueError(f"RoPE cache width must be {half}")

    if cos_cache.dtype != torch.float16:
        raise TypeError("cos_cache must use FP16(torch.float16)")

    if sin_cache.dtype != torch.float16:
        raise TypeError("sin_cache must use FP16(torch.float16)")

    if k_data.ndim != 4:
        raise ValueError("k_data/v_data must be rank-4 tensors")

    num_pages = k_data.shape[0]

    expected_data_shape = (
        num_pages,
        page_size,
        num_kv_heads,
        head_dim // 2,
    )

    expected_scale_shape = (
        num_pages,
        page_size,
        num_kv_heads,
        head_dim // group_size,
    )

    if tuple(k_data.shape) != expected_data_shape:
        raise ValueError(f"k_data must have shape {expected_data_shape}")

    if tuple(v_data.shape) != expected_data_shape:
        raise ValueError(f"v_data must have shape {expected_data_shape}")

    if tuple(k_scale.shape) != expected_scale_shape:
        raise ValueError(f"k_scale must have shape {expected_scale_shape}")

    if tuple(v_scale.shape) != expected_scale_shape:
        raise ValueError(f"v_scale must have shape {expected_scale_shape}")

    if k_data.dtype != torch.uint8:
        raise TypeError("k_data must use UINT8(torch.uint8)")

    if v_data.dtype != torch.uint8:
        raise TypeError("v_data must use UINT8(torch.uint8)")

    if k_scale.dtype != torch.float16:
        raise TypeError("k_scale must use FP16(torch.float16)")

    if v_scale.dtype != torch.float16:
        raise TypeError("v_scale must use FP16(torch.float16)")

    valid = slot_mapping >= 0

    valid_slots = slot_mapping[valid]
    valid_positions = positions[valid]

    if valid_slots.numel() > 0:
        if torch.unique(valid_slots).numel() != valid_slots.numel():
            raise ValueError("duplicate physical KV-cache slots are not allowed")

        max_slot = num_pages * page_size

        if bool(torch.any(valid_slots >= max_slot)):
            raise IndexError("slot_mapping points outside allocated KV cache")

        if bool(torch.any(valid_positions < 0)):
            raise IndexError("positions must be non-negative")

        if bool(torch.any(valid_positions >= cos_cache.shape[0])):
            raise IndexError("positions point outside RoPE cache")

    capability = torch.cuda.get_device_capability(device)

    if capability < (7, 0):
        raise RuntimeError(f"unsupported CUDA compute capability: {capability}")

    _rope_int4_paged_append(
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
        head_dim,
        page_size,
        64,
    )
