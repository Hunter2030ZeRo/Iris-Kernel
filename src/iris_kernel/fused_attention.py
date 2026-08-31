from __future__ import annotations

import torch
import tilelang
import tilelang.language as T


_LOG2_E = 1.4426950408889634


@tilelang.jit(target="cuda")
def _int4_paged_decode_attention(
    query,
    k_data,
    v_data,
    k_scale,
    v_scale,
    block_table,
    seq_lens,
    output,
    head_dim: int,
    page_size: int,
    group_size: int = 32,
    block_h: int = 16,
    block_n: int = 32,
    threads: int = 128,
    softmax_scale: float = 0.0,
):
    """Decode-only GQA attention that consumes Iris INT4 paged K/V directly."""
    B, HQ, HKV, NP, MAXP = T.const("B, HQ, HKV, NP, MAXP")

    query: T.Tensor((B, HQ, head_dim), T.float16)  # ty:ignore[invalid-type-form]
    k_data: T.Tensor(
        (NP, page_size, HKV, head_dim // 2), T.uint8
    )  # ty:ignore[invalid-type-form]
    v_data: T.Tensor(
        (NP, page_size, HKV, head_dim // 2), T.uint8
    )  # ty:ignore[invalid-type-form]
    k_scale: T.Tensor(
        (NP, page_size, HKV, head_dim // group_size), T.float16
    )  # ty:ignore[invalid-type-form]
    v_scale: T.Tensor(
        (NP, page_size, HKV, head_dim // group_size), T.float16
    )  # ty:ignore[invalid-type-form]
    block_table: T.Tensor((B, MAXP), T.int32)  # ty:ignore[invalid-type-form]
    seq_lens: T.Tensor((B,), T.int32)  # ty:ignore[invalid-type-form]
    output: T.Tensor((B, HQ, head_dim), T.float16)  # ty:ignore[invalid-type-form]

    kv_group_num = HQ // HKV
    qk_scale = head_dim**-0.5 if softmax_scale == 0.0 else softmax_scale
    exp2_scale = float(qk_scale * _LOG2_E)

    with T.Kernel(B, HKV, threads=threads) as (batch_idx, kv_head_idx):
        q_shared = T.alloc_shared((block_h, head_dim), T.float16)
        k_shared = T.alloc_shared((block_n, head_dim), T.float16)
        v_shared = T.alloc_shared((block_n, head_dim), T.float16)
        p_shared = T.alloc_shared((block_h, block_n), T.float16)

        acc_s = T.alloc_fragment((block_h, block_n), T.float32)
        acc_o = T.alloc_fragment((block_h, head_dim), T.float32)
        scores_max = T.alloc_fragment((block_h,), T.float32)
        scores_max_prev = T.alloc_fragment((block_h,), T.float32)
        scores_scale = T.alloc_fragment((block_h,), T.float32)
        scores_sum = T.alloc_fragment((block_h,), T.float32)
        logsum = T.alloc_fragment((block_h,), T.float32)

        q_head_start = kv_head_idx * kv_group_num

        for row, dim in T.Parallel(block_h, head_dim):
            safe_row = T.min(row, kv_group_num - 1)
            q_shared[row, dim] = T.if_then_else(
                row < kv_group_num,
                query[batch_idx, q_head_start + safe_row, dim],
                T.cast(0.0, T.float16),
            )

        T.clear(acc_o)
        T.fill(logsum, 0.0)
        T.fill(scores_max, -T.infinity(T.float32))

        sequence_length = seq_lens[batch_idx]
        num_logical_pages = T.ceildiv(sequence_length, page_size)
        last_logical_page = num_logical_pages - 1
        num_kv_tiles = T.ceildiv(sequence_length, block_n)

        for kv_tile in T.serial(num_kv_tiles):
            for token_in_tile, dim in T.Parallel(block_n, head_dim):
                logical_token = kv_tile * block_n + token_in_tile
                logical_page = logical_token // page_size
                safe_logical_page = T.min(logical_page, last_logical_page)
                token_in_page = logical_token % page_size
                physical_page = block_table[batch_idx, safe_logical_page]
                byte_idx = dim // 2
                group_idx = dim // group_size

                k_byte = T.cast(
                    k_data[
                        physical_page,
                        token_in_page,
                        kv_head_idx,
                        byte_idx,
                    ],
                    T.int32,
                )
                v_byte = T.cast(
                    v_data[
                        physical_page,
                        token_in_page,
                        kv_head_idx,
                        byte_idx,
                    ],
                    T.int32,
                )

                k_unsigned = T.if_then_else(
                    dim % 2 == 0,
                    k_byte & 0xF,  # ty:ignore[unsupported-operator]
                    (k_byte >> 4) & 0xF,  # ty:ignore[unsupported-operator]
                )
                v_unsigned = T.if_then_else(
                    dim % 2 == 0,
                    v_byte & 0xF,  # ty:ignore[unsupported-operator]
                    (v_byte >> 4) & 0xF,  # ty:ignore[unsupported-operator]
                )

                k_signed = T.if_then_else(
                    k_unsigned >= 8, k_unsigned - 16, k_unsigned
                )
                v_signed = T.if_then_else(
                    v_unsigned >= 8, v_unsigned - 16, v_unsigned
                )

                k_shared[token_in_tile, dim] = T.cast(
                    T.cast(k_signed, T.float32)
                    * T.cast(
                        k_scale[
                            physical_page,
                            token_in_page,
                            kv_head_idx,
                            group_idx,
                        ],
                        T.float32,
                    ),
                    T.float16,
                )
                v_shared[token_in_tile, dim] = T.cast(
                    T.cast(v_signed, T.float32)
                    * T.cast(
                        v_scale[
                            physical_page,
                            token_in_page,
                            kv_head_idx,
                            group_idx,
                        ],
                        T.float32,
                    ),
                    T.float16,
                )

            T.sync_threads()
            T.clear(acc_s)
            T.gemm(
                q_shared,
                k_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
            )

            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(T.float32))

            for row, token_in_tile in T.Parallel(block_h, block_n):
                logical_token = kv_tile * block_n + token_in_tile
                acc_s[row, token_in_tile] = T.if_then_else(
                    logical_token < sequence_length,
                    acc_s[row, token_in_tile],
                    -T.infinity(T.float32),
                )

            T.reduce_max(acc_s, scores_max, dim=1, clear=False)

            for row in T.Parallel(block_h):
                scores_max[row] = T.max(scores_max[row], scores_max_prev[row])
                scores_scale[row] = T.exp2(
                    scores_max_prev[row] * exp2_scale
                    - scores_max[row] * exp2_scale
                )

            for row, token_in_tile in T.Parallel(block_h, block_n):
                acc_s[row, token_in_tile] = T.exp2(
                    acc_s[row, token_in_tile] * exp2_scale
                    - scores_max[row] * exp2_scale
                )

            T.fill(scores_sum, 0.0)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            T.copy(acc_s, p_shared)

            for row in T.Parallel(block_h):
                logsum[row] = (
                    logsum[row] * scores_scale[row] + scores_sum[row]
                )

            for row, dim in T.Parallel(block_h, head_dim):
                acc_o[row, dim] *= scores_scale[row]

            T.gemm(
                p_shared,
                v_shared,
                acc_o,
                policy=T.GemmWarpPolicy.FullCol,
            )
            T.sync_threads()

        for row, dim in T.Parallel(block_h, head_dim):
            if row < kv_group_num:
                output[batch_idx, q_head_start + row, dim] = T.cast(
                    acc_o[row, dim] / logsum[row], T.float16
                )


@tilelang.jit(target="cuda")
def _fp16_paged_decode_attention(
    query,
    k_cache,
    v_cache,
    block_table,
    seq_lens,
    output,
    head_dim: int,
    page_size: int,
    block_h: int = 16,
    block_n: int = 32,
    threads: int = 128,
    softmax_scale: float = 0.0,
):
    """Matched FP16 paged decode-attention baseline using the same schedule."""
    B, HQ, HKV, NP, MAXP = T.const("B, HQ, HKV, NP, MAXP")

    query: T.Tensor((B, HQ, head_dim), T.float16)  # ty:ignore[invalid-type-form]
    k_cache: T.Tensor(
        (NP, page_size, HKV, head_dim), T.float16
    )  # ty:ignore[invalid-type-form]
    v_cache: T.Tensor(
        (NP, page_size, HKV, head_dim), T.float16
    )  # ty:ignore[invalid-type-form]
    block_table: T.Tensor((B, MAXP), T.int32)  # ty:ignore[invalid-type-form]
    seq_lens: T.Tensor((B,), T.int32)  # ty:ignore[invalid-type-form]
    output: T.Tensor((B, HQ, head_dim), T.float16)  # ty:ignore[invalid-type-form]

    kv_group_num = HQ // HKV
    qk_scale = head_dim**-0.5 if softmax_scale == 0.0 else softmax_scale
    exp2_scale = float(qk_scale * _LOG2_E)

    with T.Kernel(B, HKV, threads=threads) as (batch_idx, kv_head_idx):
        q_shared = T.alloc_shared((block_h, head_dim), T.float16)
        k_shared = T.alloc_shared((block_n, head_dim), T.float16)
        v_shared = T.alloc_shared((block_n, head_dim), T.float16)
        p_shared = T.alloc_shared((block_h, block_n), T.float16)

        acc_s = T.alloc_fragment((block_h, block_n), T.float32)
        acc_o = T.alloc_fragment((block_h, head_dim), T.float32)
        scores_max = T.alloc_fragment((block_h,), T.float32)
        scores_max_prev = T.alloc_fragment((block_h,), T.float32)
        scores_scale = T.alloc_fragment((block_h,), T.float32)
        scores_sum = T.alloc_fragment((block_h,), T.float32)
        logsum = T.alloc_fragment((block_h,), T.float32)

        q_head_start = kv_head_idx * kv_group_num

        for row, dim in T.Parallel(block_h, head_dim):
            safe_row = T.min(row, kv_group_num - 1)
            q_shared[row, dim] = T.if_then_else(
                row < kv_group_num,
                query[batch_idx, q_head_start + safe_row, dim],
                T.cast(0.0, T.float16),
            )

        T.clear(acc_o)
        T.fill(logsum, 0.0)
        T.fill(scores_max, -T.infinity(T.float32))

        sequence_length = seq_lens[batch_idx]
        num_logical_pages = T.ceildiv(sequence_length, page_size)
        last_logical_page = num_logical_pages - 1
        num_kv_tiles = T.ceildiv(sequence_length, block_n)

        for kv_tile in T.serial(num_kv_tiles):
            for token_in_tile, dim in T.Parallel(block_n, head_dim):
                logical_token = kv_tile * block_n + token_in_tile
                logical_page = logical_token // page_size
                safe_logical_page = T.min(logical_page, last_logical_page)
                token_in_page = logical_token % page_size
                physical_page = block_table[batch_idx, safe_logical_page]

                k_shared[token_in_tile, dim] = k_cache[
                    physical_page,
                    token_in_page,
                    kv_head_idx,
                    dim,
                ]
                v_shared[token_in_tile, dim] = v_cache[
                    physical_page,
                    token_in_page,
                    kv_head_idx,
                    dim,
                ]

            T.sync_threads()
            T.clear(acc_s)
            T.gemm(
                q_shared,
                k_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
            )

            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(T.float32))

            for row, token_in_tile in T.Parallel(block_h, block_n):
                logical_token = kv_tile * block_n + token_in_tile
                acc_s[row, token_in_tile] = T.if_then_else(
                    logical_token < sequence_length,
                    acc_s[row, token_in_tile],
                    -T.infinity(T.float32),
                )

            T.reduce_max(acc_s, scores_max, dim=1, clear=False)

            for row in T.Parallel(block_h):
                scores_max[row] = T.max(scores_max[row], scores_max_prev[row])
                scores_scale[row] = T.exp2(
                    scores_max_prev[row] * exp2_scale
                    - scores_max[row] * exp2_scale
                )

            for row, token_in_tile in T.Parallel(block_h, block_n):
                acc_s[row, token_in_tile] = T.exp2(
                    acc_s[row, token_in_tile] * exp2_scale
                    - scores_max[row] * exp2_scale
                )

            T.fill(scores_sum, 0.0)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            T.copy(acc_s, p_shared)

            for row in T.Parallel(block_h):
                logsum[row] = (
                    logsum[row] * scores_scale[row] + scores_sum[row]
                )

            for row, dim in T.Parallel(block_h, head_dim):
                acc_o[row, dim] *= scores_scale[row]

            T.gemm(
                p_shared,
                v_shared,
                acc_o,
                policy=T.GemmWarpPolicy.FullCol,
            )
            T.sync_threads()

        for row, dim in T.Parallel(block_h, head_dim):
            if row < kv_group_num:
                output[batch_idx, q_head_start + row, dim] = T.cast(
                    acc_o[row, dim] / logsum[row], T.float16
                )


def _validate_common_decode_inputs(
    query: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_pages: int,
    num_kv_heads: int,
    page_size: int,
) -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("Iris fused kernels require CUDA")
    if query.device.type != "cuda":
        raise ValueError("query must be a CUDA tensor")
    if query.ndim != 3:
        raise ValueError("query must have shape [B, Hq, D]")
    if query.dtype != torch.float16:
        raise TypeError("query must use FP16(torch.float16)")
    if page_size != 16:
        raise NotImplementedError("decode fast path currently supports page_size=16")

    batch, num_query_heads, head_dim = query.shape
    if head_dim not in (64, 128):
        raise NotImplementedError(
            "decode fast path currently supports head_dim in {64, 128}"
        )
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if num_query_heads // num_kv_heads > 16:
        raise NotImplementedError(
            "decode fast path supports at most 16 query heads per KV head"
        )

    if block_table.ndim != 2 or block_table.shape[0] != batch:
        raise ValueError("block_table must have shape [B, max_pages_per_sequence]")
    if seq_lens.shape != (batch,):
        raise ValueError("seq_lens must have shape [B]")
    if block_table.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise TypeError("block_table and seq_lens must use torch.int32")

    device = query.device
    if block_table.device != device or seq_lens.device != device:
        raise ValueError("block_table and seq_lens must be on the query device")
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
        pages = block_table[batch_idx, :used]
        if bool(torch.any(pages < 0)) or bool(torch.any(pages >= num_pages)):
            raise IndexError("block_table contains an invalid physical page id")

    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 0):
        raise RuntimeError(f"unsupported CUDA compute capability: {capability}")
    return batch, num_query_heads, head_dim


def fused_int4_paged_decode_attention(
    query: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor | None = None,
    *,
    page_size: int = 16,
    group_size: int = 32,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """
    Fused decode attention over Iris INT4 paged K/V.

    Direct path: packed load -> signed nibble unpack -> group dequantization
    -> QK -> online softmax -> PV. No FP16 global K/V materialization.
    """
    if group_size != 32:
        raise NotImplementedError("decode fast path currently supports group_size=32")
    if k_data.ndim != 4 or tuple(v_data.shape) != tuple(k_data.shape):
        raise ValueError("k_data/v_data must have identical rank-4 shapes")

    num_pages, cache_page_size, num_kv_heads, packed_dim = k_data.shape
    _, _, head_dim = query.shape
    if cache_page_size != page_size or packed_dim * 2 != head_dim:
        raise ValueError("packed cache shape does not match query/page_size")

    _validate_common_decode_inputs(
        query,
        block_table,
        seq_lens,
        num_pages=num_pages,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
    )

    expected_scale_shape = (
        num_pages,
        page_size,
        num_kv_heads,
        head_dim // group_size,
    )
    if tuple(k_scale.shape) != expected_scale_shape:
        raise ValueError(f"k_scale must have shape {expected_scale_shape}")
    if tuple(v_scale.shape) != expected_scale_shape:
        raise ValueError(f"v_scale must have shape {expected_scale_shape}")
    if k_data.dtype != torch.uint8 or v_data.dtype != torch.uint8:
        raise TypeError("k_data/v_data must use UINT8(torch.uint8)")
    if k_scale.dtype != torch.float16 or v_scale.dtype != torch.float16:
        raise TypeError("k_scale/v_scale must use FP16(torch.float16)")

    device = query.device
    for name, tensor in (
        ("k_data", k_data),
        ("v_data", v_data),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")

    if output is None:
        output = torch.empty_like(query)
    elif output.shape != query.shape or output.dtype != query.dtype:
        raise ValueError("output must match query shape and dtype")
    elif output.device != query.device:
        raise ValueError("output must be on the same device as query")

    _int4_paged_decode_attention(
        query,
        k_data,
        v_data,
        k_scale,
        v_scale,
        block_table,
        seq_lens,
        output,
        head_dim,
        page_size,
        group_size,
        16,
        32,
        128,
        0.0 if softmax_scale is None else float(softmax_scale),
    )
    return output


def fused_fp16_paged_decode_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor | None = None,
    *,
    page_size: int = 16,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Matched FP16 paged decode-attention baseline using the same schedule."""
    if k_cache.ndim != 4 or tuple(v_cache.shape) != tuple(k_cache.shape):
        raise ValueError("k_cache/v_cache must have identical rank-4 shapes")

    num_pages, cache_page_size, num_kv_heads, cache_dim = k_cache.shape
    _, _, head_dim = query.shape
    if cache_page_size != page_size or cache_dim != head_dim:
        raise ValueError("FP16 cache shape does not match query/page_size")

    _validate_common_decode_inputs(
        query,
        block_table,
        seq_lens,
        num_pages=num_pages,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
    )
    if k_cache.dtype != torch.float16 or v_cache.dtype != torch.float16:
        raise TypeError("k_cache/v_cache must use FP16")
    if k_cache.device != query.device or v_cache.device != query.device:
        raise ValueError("k_cache/v_cache must be on the query device")

    if output is None:
        output = torch.empty_like(query)
    elif output.shape != query.shape or output.dtype != query.dtype:
        raise ValueError("output must match query shape and dtype")
    elif output.device != query.device:
        raise ValueError("output must be on the same device as query")

    _fp16_paged_decode_attention(
        query,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        output,
        head_dim,
        page_size,
        16,
        32,
        128,
        0.0 if softmax_scale is None else float(softmax_scale),
    )
    return output
