from __future__ import annotations

import math

import pytest
import torch

from iris_kernel.fused_attention import fused_int4_paged_decode_attention
from iris_kernel.torch_eager_attention import (
    torch_eager_fp16_paged_decode_attention,
    torch_eager_int4_paged_decode_attention,
)
from iris_kernel.torch_eager_kv import apply_rope, torch_eager_append


def _make_attention_case(
    *,
    batch: int = 2,
    num_query_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    page_size: int = 16,
    group_size: int = 32,
):
    torch.manual_seed(3407)
    device = torch.device("cuda")

    seq_lens = torch.tensor([23, 31], device=device, dtype=torch.int32)
    max_seq_len = int(seq_lens.max().item())
    max_pages = math.ceil(max_seq_len / page_size)
    num_pages = batch * max_pages + 3

    page_ids = torch.randperm(num_pages, device=device, dtype=torch.int64)[
        : batch * max_pages
    ].reshape(batch, max_pages)
    block_table = page_ids.to(torch.int32)

    position_ids = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
            / head_dim
        )
    )
    freqs = position_ids[:, None] * inv_freq[None, :]
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
    k_data = torch.zeros(data_shape, device=device, dtype=torch.uint8)
    v_data = torch.zeros_like(k_data)
    k_scale = torch.zeros(scale_shape, device=device, dtype=torch.float16)
    v_scale = torch.zeros_like(k_scale)

    fp16_k_cache = torch.zeros(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    fp16_v_cache = torch.zeros_like(fp16_k_cache)

    flat_key = []
    flat_value = []
    flat_positions = []
    flat_slots = []

    for batch_idx in range(batch):
        seq_len = int(seq_lens[batch_idx].item())
        key = torch.randn(
            seq_len,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.float16,
        )
        value = torch.randn_like(key)
        positions = torch.arange(seq_len, device=device, dtype=torch.int32)

        physical_slots = []
        for logical_token in range(seq_len):
            physical_page = int(
                block_table[batch_idx, logical_token // page_size].item()
            )
            offset = logical_token % page_size
            physical_slots.append(physical_page * page_size + offset)

        slots = torch.tensor(physical_slots, device=device, dtype=torch.int32)
        flat_key.append(key)
        flat_value.append(value)
        flat_positions.append(positions)
        flat_slots.append(slots)

        rotated_key = apply_rope(
            key,
            cos_cache[positions.long()][:, None, :],
            sin_cache[positions.long()][:, None, :],
            rotary_dim=head_dim,
        )
        pages = slots.long() // page_size
        offsets = slots.long() % page_size
        fp16_k_cache[pages, offsets] = rotated_key
        fp16_v_cache[pages, offsets] = value

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
    query_positions = (seq_lens - 1).long()
    query = apply_rope(
        query_unrotated,
        cos_cache[query_positions][:, None, :],
        sin_cache[query_positions][:, None, :],
        rotary_dim=head_dim,
    )

    return {
        "query": query,
        "k_data": k_data,
        "v_data": v_data,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "fp16_k_cache": fp16_k_cache,
        "fp16_v_cache": fp16_v_cache,
        "page_size": page_size,
        "group_size": group_size,
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Iris fused attention requires CUDA",
)
def test_fused_int4_paged_decode_attention_matches_eager():
    case = _make_attention_case()

    expected = torch_eager_int4_paged_decode_attention(
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
    actual = fused_int4_paged_decode_attention(
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
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_int4_attention_tracks_fp16_attention():
    case = _make_attention_case()

    int4_output = torch_eager_int4_paged_decode_attention(
        case["query"],
        case["k_data"],
        case["v_data"],
        case["k_scale"],
        case["v_scale"],
        case["block_table"],
        case["seq_lens"],
        page_size=case["page_size"],
        group_size=case["group_size"],
    ).float()
    fp16_output = torch_eager_fp16_paged_decode_attention(
        case["query"],
        case["fp16_k_cache"],
        case["fp16_v_cache"],
        case["block_table"],
        case["seq_lens"],
        page_size=case["page_size"],
    ).float()

    cosine = torch.nn.functional.cosine_similarity(
        int4_output.flatten(0, 1), fp16_output.flatten(0, 1), dim=-1
    ).mean()
    rmse = torch.mean((int4_output - fp16_output) ** 2).sqrt()

    assert float(cosine) > 0.97
    assert float(rmse) < 0.20
