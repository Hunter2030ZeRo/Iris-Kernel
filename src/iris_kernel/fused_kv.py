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
        raise RuntimeError(f"unsupported CUDA capability: {capability}")

    # The kernel itself handles slot < 0.
    #
    # Other metadata validation stays on the host.
    valid_slots = slot_mapping[slot_mapping >= 0]

    if valid_slots.numel() > 0:
        if torch.unique(valid_slots).numel() != valid_slots.numel():
            raise ValueError("duplicate physical KV-cache slots are not allowed")

        max_slot = k_cache.shape[0] * page_size

        if bool(torch.any(valid_slots >= max_slot)):
            raise IndexError("slot_mapping points outside allocated cache")

    if bool(torch.any(positions < 0)):
        raise IndexError("positions must be non-negative")

    if bool(torch.any(positions >= cos_cache.shape[0])):
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
