from __future__ import annotations

import torch


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:

    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")

    if rotary_dim < 0:
        raise ValueError("rotary_dim must be non-negative")

    if rotary_dim > x.shape[-1]:
        raise ValueError(
            f"rotary_dim ({rotary_dim}) cannot exceed head_dim ({x.shape[-1]})"
        )

    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")

    if rotary_dim == 0:
        return x

    half = rotary_dim // 2

    if cos.shape[-1] != half or sin.shape[-1] != half:
        raise ValueError(f"cos/sin width must be rotary_dim // 2 ({half})")

    rotary = x[..., :rotary_dim]
    passthrough = x[..., rotary_dim:]

    x1 = rotary[..., :half]
    x2 = rotary[..., half:]

    rotated = torch.cat(
        (
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ),
        dim=-1,
    )

    return torch.cat((rotated, passthrough), dim=-1)


def quantize_int4_groupwise(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    if group_size <= 0:
        raise ValueError("group_size must be positive")

    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"last dimension ({x.shape[-1]}) must be divisible "
            f"by group_size ({group_size})"
        )

    *prefix, dim = x.shape
    groups = dim // group_size

    grouped = x.float().reshape(
        *prefix,
        groups,
        group_size,
    )

    amax = grouped.abs().amax(dim=-1)

    min_scale = torch.finfo(x.dtype).tiny

    scale32 = (amax / 7.0).clamp_min(min_scale)

    scales = scale32.to(x.dtype)

    quant_scale = scales.float()

    q = torch.round(grouped / quant_scale[..., None])
    q = q.clamp(-7, 7).to(torch.int8)

    return q.reshape_as(x), scales


def pack_int4(q: torch.Tensor) -> torch.Tensor:

    if q.dtype != torch.int8:
        raise TypeError(f"q must have dtype torch.int8, got {q.dtype}")

    if q.shape[-1] % 2 != 0:
        raise ValueError("last dimension must be even for INT4 packing")

    if torch.any(q < -7) or torch.any(q > 7):
        raise ValueError("INT4 values must lie in [-7, 7]")

    q16 = q.to(torch.int16)

    lo = q16[..., 0::2] & 0xF
    hi = q16[..., 1::2] & 0xF

    return (lo | (hi << 4)).to(torch.uint8)


@torch.no_grad()
def torch_eager_append(
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
    group_size: int,
    page_size: int,
) -> None:

    if key.shape != value.shape:
        raise ValueError("key and value must have identical shapes")

    if key.ndim != 3:
        raise ValueError("key/value must have shape [T, Hkv, D]")

    if key.dtype != value.dtype:
        raise TypeError("key and value must have identical dtypes")

    if not key.is_floating_point():
        raise TypeError("key and value must use a floating-point dtype")

    num_tokens, num_kv_heads, head_dim = key.shape

    if positions.shape != (num_tokens,):
        raise ValueError(
            f"positions must have shape ({num_tokens},), got {tuple(positions.shape)}"
        )

    if slot_mapping.shape != (num_tokens,):
        raise ValueError(
            f"slot_mapping must have shape ({num_tokens},), "
            f"got {tuple(slot_mapping.shape)}"
        )

    if rotary_dim < 0:
        raise ValueError("rotary_dim must be non-negative")

    if rotary_dim > head_dim:
        raise ValueError("rotary_dim cannot exceed head_dim")

    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")

    if group_size <= 0:
        raise ValueError("group_size must be positive")

    if page_size <= 0:
        raise ValueError("page_size must be positive")

    if head_dim % group_size != 0:
        raise ValueError("head_dim must be divisible by group_size")

    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for INT4 packing")

    device = key.device

    for name, tensor in (
        ("value", value),
        ("cos_cache", cos_cache),
        ("sin_cache", sin_cache),
        ("k_data", k_data),
        ("v_data", v_data),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")

    if cos_cache.shape != sin_cache.shape:
        raise ValueError("cos_cache and sin_cache must have identical shapes")

    if cos_cache.ndim != 2:
        raise ValueError(
            "cos_cache/sin_cache must have shape [max_position, rotary_dim // 2]"
        )

    expected_rope_width = rotary_dim // 2

    if cos_cache.shape[-1] != expected_rope_width:
        raise ValueError(
            f"RoPE cache width must be {expected_rope_width}, got {cos_cache.shape[-1]}"
        )

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
        raise ValueError(
            f"k_data shape mismatch: expected "
            f"{expected_data_shape}, "
            f"got {tuple(k_data.shape)}"
        )

    if tuple(v_data.shape) != expected_data_shape:
        raise ValueError(
            f"v_data shape mismatch: expected "
            f"{expected_data_shape}, "
            f"got {tuple(v_data.shape)}"
        )

    if tuple(k_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"k_scale shape mismatch: expected "
            f"{expected_scale_shape}, "
            f"got {tuple(k_scale.shape)}"
        )

    if tuple(v_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"v_scale shape mismatch: expected "
            f"{expected_scale_shape}, "
            f"got {tuple(v_scale.shape)}"
        )

    if k_data.dtype != torch.uint8:
        raise TypeError(f"k_data must use torch.uint8, got {k_data.dtype}")

    if v_data.dtype != torch.uint8:
        raise TypeError(f"v_data must use torch.uint8, got {v_data.dtype}")

    if k_scale.dtype != key.dtype:
        raise TypeError(
            f"k_scale dtype must match key dtype ({key.dtype}), got {k_scale.dtype}"
        )

    if v_scale.dtype != key.dtype:
        raise TypeError(
            f"v_scale dtype must match value dtype ({value.dtype}), got {v_scale.dtype}"
        )

    positions = positions.to(
        device=device,
        dtype=torch.int64,
    )

    slot_mapping = slot_mapping.to(
        device=device,
        dtype=torch.int64,
    )

    valid = slot_mapping >= 0

    if not bool(torch.any(valid)):
        return

    slots = slot_mapping[valid]
    pos = positions[valid]

    key_valid = key[valid]
    value_valid = value[valid]

    if torch.unique(slots).numel() != slots.numel():
        raise ValueError("duplicate physical KV-cache slots are not allowed")

    if bool(torch.any(pos < 0)):
        raise IndexError("positions must be non-negative")

    if bool(torch.any(pos >= cos_cache.shape[0])):
        raise IndexError("positions point outside RoPE cache")

    pages = torch.div(
        slots,
        page_size,
        rounding_mode="floor",
    )

    offsets = slots % page_size

    if bool(torch.any(pages >= num_pages)):
        raise IndexError("slot_mapping points outside allocated KV cache")

    cos = cos_cache[pos].to(dtype=key.dtype)
    sin = sin_cache[pos].to(dtype=key.dtype)

    cos = cos[:, None, :]
    sin = sin[:, None, :]

    rotated_key = apply_rope(
        key_valid,
        cos,
        sin,
        rotary_dim=rotary_dim,
    )

    key_q, key_s = quantize_int4_groupwise(
        rotated_key,
        group_size=group_size,
    )

    value_q, value_s = quantize_int4_groupwise(
        value_valid,
        group_size=group_size,
    )

    key_packed = pack_int4(key_q)
    value_packed = pack_int4(value_q)

    k_data[pages, offsets] = key_packed
    v_data[pages, offsets] = value_packed

    k_scale[pages, offsets] = key_s
    v_scale[pages, offsets] = value_s
