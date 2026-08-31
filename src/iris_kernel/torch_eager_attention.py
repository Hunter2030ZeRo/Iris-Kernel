from __future__ import annotations

import math

import torch


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack Iris low-nibble-first signed INT4 into int8 values."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must use torch.uint8, got {packed.dtype}")

    low = (packed & 0x0F).to(torch.int16)
    high = ((packed >> 4) & 0x0F).to(torch.int16)

    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    output = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        device=packed.device,
        dtype=torch.int8,
    )
    output[..., 0::2] = low.to(torch.int8)
    output[..., 1::2] = high.to(torch.int8)
    return output


def dequantize_int4_groupwise(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = 32,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize Iris packed INT4 values using per-group scales."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    q = unpack_int4(packed).float()
    head_dim = q.shape[-1]

    if head_dim % group_size != 0:
        raise ValueError("head_dim must be divisible by group_size")

    expected_scale_shape = (*q.shape[:-1], head_dim // group_size)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"scales must have shape {expected_scale_shape}, got {tuple(scales.shape)}"
        )

    expanded_scales = scales.float().repeat_interleave(group_size, dim=-1)
    return (q * expanded_scales).to(dtype)


def _validate_attention_inputs(
    query: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    group_size: int,
) -> tuple[int, int, int, int, int]:
    if query.ndim != 3:
        raise ValueError("query must have shape [B, Hq, D]")
    if query.dtype != torch.float16:
        raise TypeError("query must use torch.float16")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    batch, num_query_heads, head_dim = query.shape

    if k_data.ndim != 4:
        raise ValueError("k_data/v_data must have shape [NP, P, Hkv, D/2]")
    if tuple(v_data.shape) != tuple(k_data.shape):
        raise ValueError("k_data and v_data must have identical shapes")
    if k_data.dtype != torch.uint8 or v_data.dtype != torch.uint8:
        raise TypeError("k_data/v_data must use torch.uint8")

    num_pages, cache_page_size, num_kv_heads, packed_dim = k_data.shape
    if cache_page_size != page_size:
        raise ValueError("cache page dimension does not match page_size")
    if packed_dim * 2 != head_dim:
        raise ValueError("packed cache head dimension does not match query")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if head_dim % group_size != 0:
        raise ValueError("head_dim must be divisible by group_size")

    expected_scale_shape = (
        num_pages,
        page_size,
        num_kv_heads,
        head_dim // group_size,
    )
    if tuple(k_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"k_scale must have shape {expected_scale_shape}, got {tuple(k_scale.shape)}"
        )
    if tuple(v_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"v_scale must have shape {expected_scale_shape}, got {tuple(v_scale.shape)}"
        )
    if k_scale.dtype != torch.float16 or v_scale.dtype != torch.float16:
        raise TypeError("k_scale/v_scale must use torch.float16")

    if block_table.ndim != 2 or block_table.shape[0] != batch:
        raise ValueError("block_table must have shape [B, max_pages_per_sequence]")
    if seq_lens.shape != (batch,):
        raise ValueError("seq_lens must have shape [B]")

    device = query.device
    for name, tensor in (
        ("k_data", k_data),
        ("v_data", v_data),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
        ("block_table", block_table),
        ("seq_lens", seq_lens),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")

    if block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("block_table must use int32 or int64")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError("seq_lens must use int32 or int64")
    if bool(torch.any(seq_lens <= 0)):
        raise ValueError("all sequence lengths must be positive")

    required_pages = torch.div(
        seq_lens.to(torch.int64) + page_size - 1,
        page_size,
        rounding_mode="floor",
    )
    if bool(torch.any(required_pages > block_table.shape[1])):
        raise ValueError("block_table does not contain enough pages")

    for batch_idx in range(batch):
        used = int(required_pages[batch_idx].item())
        page_ids = block_table[batch_idx, :used]
        if bool(torch.any(page_ids < 0)) or bool(torch.any(page_ids >= num_pages)):
            raise IndexError("block_table contains an invalid physical page id")

    return batch, num_query_heads, num_kv_heads, head_dim, num_pages


@torch.no_grad()
def torch_eager_int4_paged_decode_attention(
    query: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int = 16,
    group_size: int = 32,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """
    Read packed paged INT4 K/V, dequantize, and execute GQA decode attention.

    This is an intentionally readable PyTorch eager reference. Query shape is
    [batch, query_heads, head_dim], i.e. one decode query token per sequence.
    """
    batch, num_query_heads, num_kv_heads, head_dim, _ = _validate_attention_inputs(
        query,
        k_data,
        v_data,
        k_scale,
        v_scale,
        block_table,
        seq_lens,
        page_size=page_size,
        group_size=group_size,
    )

    scale = float(head_dim**-0.5 if softmax_scale is None else softmax_scale)
    heads_per_kv = num_query_heads // num_kv_heads
    output = torch.empty_like(query)

    for batch_idx in range(batch):
        seq_len = int(seq_lens[batch_idx].item())
        logical_pages = math.ceil(seq_len / page_size)
        page_ids = block_table[batch_idx, :logical_pages].to(torch.int64)

        k_packed = k_data[page_ids].reshape(
            logical_pages * page_size,
            num_kv_heads,
            head_dim // 2,
        )[:seq_len]
        v_packed = v_data[page_ids].reshape(
            logical_pages * page_size,
            num_kv_heads,
            head_dim // 2,
        )[:seq_len]
        k_scales = k_scale[page_ids].reshape(
            logical_pages * page_size,
            num_kv_heads,
            head_dim // group_size,
        )[:seq_len]
        v_scales = v_scale[page_ids].reshape(
            logical_pages * page_size,
            num_kv_heads,
            head_dim // group_size,
        )[:seq_len]

        key = dequantize_int4_groupwise(
            k_packed,
            k_scales,
            group_size=group_size,
            dtype=torch.float32,
        ).transpose(0, 1)
        value = dequantize_int4_groupwise(
            v_packed,
            v_scales,
            group_size=group_size,
            dtype=torch.float32,
        ).transpose(0, 1)

        key = key.repeat_interleave(heads_per_kv, dim=0)
        value = value.repeat_interleave(heads_per_kv, dim=0)
        q = query[batch_idx].float()

        logits = torch.einsum("hd,hld->hl", q, key) * scale
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        output[batch_idx] = torch.einsum("hl,hld->hd", probabilities, value).to(
            query.dtype
        )

    return output


@torch.no_grad()
def torch_eager_fp16_paged_decode_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int = 16,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """FP16 paged GQA decode-attention reference for quality comparisons."""
    if query.ndim != 3:
        raise ValueError("query must have shape [B, Hq, D]")
    if k_cache.ndim != 4 or tuple(k_cache.shape) != tuple(v_cache.shape):
        raise ValueError("k_cache/v_cache must have identical rank-4 shapes")

    batch, num_query_heads, head_dim = query.shape
    num_pages, cache_page_size, num_kv_heads, cache_dim = k_cache.shape
    if cache_page_size != page_size or cache_dim != head_dim:
        raise ValueError("FP16 cache shape does not match query/page_size")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if block_table.shape[0] != batch or seq_lens.shape != (batch,):
        raise ValueError("invalid block_table or seq_lens shape")

    scale = float(head_dim**-0.5 if softmax_scale is None else softmax_scale)
    heads_per_kv = num_query_heads // num_kv_heads
    output = torch.empty_like(query)

    for batch_idx in range(batch):
        seq_len = int(seq_lens[batch_idx].item())
        logical_pages = math.ceil(seq_len / page_size)
        page_ids = block_table[batch_idx, :logical_pages].to(torch.int64)
        if bool(torch.any(page_ids < 0)) or bool(torch.any(page_ids >= num_pages)):
            raise IndexError("block_table contains an invalid page id")

        key = k_cache[page_ids].reshape(
            logical_pages * page_size, num_kv_heads, head_dim
        )[:seq_len].transpose(0, 1).float()
        value = v_cache[page_ids].reshape(
            logical_pages * page_size, num_kv_heads, head_dim
        )[:seq_len].transpose(0, 1).float()
        key = key.repeat_interleave(heads_per_kv, dim=0)
        value = value.repeat_interleave(heads_per_kv, dim=0)

        logits = torch.einsum("hd,hld->hl", query[batch_idx].float(), key) * scale
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        output[batch_idx] = torch.einsum(
            "hl,hld->hd", probabilities, value
        ).to(query.dtype)

    return output
