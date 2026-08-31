import pytest
import torch

from iris_kernel.torch_eager_kv import (
    apply_rope,
    quantize_int4_groupwise,
    pack_int4,
    torch_eager_append,
)

from iris_kernel.fused_kv import fused_rope_fp16_append


def test_rope_pos_zero_id():
    x = torch.randn(2, 3, 8)

    cos = torch.ones(2, 1, 4)
    sin = torch.zeros(2, 1, 4)

    out = apply_rope(
        x,
        cos,
        sin,
        rotary_dim=8,
    )

    torch.testing.assert_close(out, x)


def test_rope_known_quarter_turn():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    cos = torch.zeros(1, 1, 2)
    sin = torch.ones(1, 1, 2)

    out = apply_rope(
        x,
        cos,
        sin,
        rotary_dim=4,
    )

    expected = torch.tensor([[[-3.0, -4.0, 1.0, 2.0]]])

    torch.testing.assert_close(out, expected)


def test_quantize_int4_known_values():
    x = torch.tensor([[[-7.0, -3.0, 0.0, 7.0]]])

    q, scales = quantize_int4_groupwise(x, group_size=4)

    expected_q = torch.tensor(
        [[[-7, -3, 0, 7]]],
        dtype=torch.int8,
    )

    expected_scale = torch.tensor(
        [[[1.0]]],
        dtype=x.dtype,
    )

    torch.testing.assert_close(q, expected_q)
    torch.testing.assert_close(scales, expected_scale)

    assert q.dtype == torch.int8
    assert q.min().item() >= -7
    assert q.max().item() <= 7


def test_quantize_int4_zero_group_is_finite():
    x = torch.zeros(2, 3, 8)

    q, scales = quantize_int4_groupwise(x, group_size=4)

    assert torch.all(q == 0)
    assert torch.all(torch.isfinite(scales))
    assert torch.all(scales > 0)


def test_pack_int4_known_values():
    q = torch.tensor(
        [[-7, -1, 0, 1, 6, 7]],
        dtype=torch.int8,
    )

    packed = pack_int4(q)

    expected = torch.tensor(
        [[0xF9, 0x10, 0x76]],
        dtype=torch.uint8,
    )

    torch.testing.assert_close(packed, expected)

    assert packed.dtype == torch.uint8
    assert packed.shape[-1] == q.shape[-1] // 2


def _make_append_case():
    torch.manual_seed(3407)

    num_tokens = 4
    num_kv_heads = 2
    head_dim = 8

    page_size = 2
    num_pages = 4
    group_size = 4
    rotary_dim = 8

    key = torch.randn(
        num_tokens,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
    )

    value = torch.rand_like(key)

    positions = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.int64,
    )

    slot_mapping = torch.tensor(
        [3, 0, -1, 6],
        dtype=torch.int64,
    )

    cos_cache = torch.ones(
        4,
        rotary_dim // 2,
        dtype=torch.float16,
    )
    sin_cache = torch.zeros_like(cos_cache)

    k_data = torch.randint(
        0,
        256,
        (
            num_pages,
            page_size,
            num_kv_heads,
            head_dim // 2,
        ),
        dtype=torch.uint8,
    )
    v_data = torch.randint(
        0,
        256,
        k_data.shape,
        dtype=torch.uint8,
    )

    k_scale = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim // group_size,
        dtype=torch.float16,
    )
    v_scale = torch.randn_like(k_scale)

    return {
        "key": key,
        "value": value,
        "positions": positions,
        "slot_mapping": slot_mapping,
        "cos_cache": cos_cache,
        "sin_cache": sin_cache,
        "k_data": k_data,
        "v_data": v_data,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "rotary_dim": rotary_dim,
        "group_size": group_size,
        "page_size": page_size,
    }


def test_paged_append_scattered_slots():
    case = _make_append_case()

    torch_eager_append(**case)

    valid = case["slot_mapping"] >= 0

    slots = case["slot_mapping"][valid]
    pages = slots // case["page_size"]
    offsets = slots % case["page_size"]

    expected_k_q, expected_k_scale = quantize_int4_groupwise(
        case["key"][valid],
        group_size=case["group_size"],
    )

    expected_v_q, expected_v_scale = quantize_int4_groupwise(
        case["value"][valid],
        group_size=case["group_size"],
    )

    expected_k = pack_int4(expected_k_q)
    expected_v = pack_int4(expected_v_q)

    torch.testing.assert_close(
        case["k_data"][pages, offsets],
        expected_k,
    )

    torch.testing.assert_close(
        case["v_data"][pages, offsets],
        expected_v,
    )

    torch.testing.assert_close(
        case["k_scale"][pages, offsets],
        expected_k_scale,
    )

    torch.testing.assert_close(
        case["v_scale"][pages, offsets],
        expected_v_scale,
    )


def test_unwritten_cache_regions_are_unchanged():
    case = _make_append_case()

    k_before = case["k_data"].clone()
    v_before = case["v_data"].clone()
    ks_before = case["k_scale"].clone()
    vs_before = case["v_scale"].clone()

    torch_eager_append(**case)

    mask = torch.zeros(
        case["k_data"].shape[:2],
        dtype=torch.bool,
    )

    valid_slots = case["slot_mapping"][case["slot_mapping"] >= 0]

    pages = valid_slots // case["page_size"]
    offsets = valid_slots % case["page_size"]

    mask[pages, offsets] = True

    assert torch.equal(
        case["k_data"][~mask],
        k_before[~mask],
    )

    assert torch.equal(
        case["v_data"][~mask],
        v_before[~mask],
    )

    assert torch.equal(
        case["k_scale"][~mask],
        ks_before[~mask],
    )

    assert torch.equal(
        case["v_scale"][~mask],
        vs_before[~mask],
    )


def test_negative_slot_is_ignored():
    case = _make_append_case()

    case["slot_mapping"] = torch.full_like(
        case["slot_mapping"],
        -1,
    )

    k_before = case["k_data"].clone()
    v_before = case["v_data"].clone()
    ks_before = case["k_scale"].clone()
    vs_before = case["v_scale"].clone()

    torch_eager_append(**case)

    assert torch.equal(case["k_data"], k_before)
    assert torch.equal(case["v_data"], v_before)
    assert torch.equal(case["k_scale"], ks_before)
    assert torch.equal(case["v_scale"], vs_before)


def test_duplicate_slots_rejected():
    case = _make_append_case()

    case["slot_mapping"] = torch.tensor(
        [3, 0, 3, 6],
        dtype=torch.int64,
    )

    with pytest.raises(ValueError):
        torch_eager_append(**case)


def test_rope_preserves_passthrough_dims():
    x = torch.randn(2, 3, 8)

    rotary_dim = 4
    cos = torch.zeros(2, 1, 2)
    sin = torch.ones(2, 1, 2)

    out = apply_rope(
        x,
        cos,
        sin,
        rotary_dim=rotary_dim,
    )

    torch.testing.assert_close(
        out[..., rotary_dim:],
        x[..., rotary_dim:],
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required for TileLang kernel"
)
def test_fused_rope_append_matches_eager_ref():
    torch.manual_seed(3407)

    device = torch.device("cuda")

    num_tokens = 7
    num_kv_heads = 4
    head_dim = 128

    rotary_dim = 128
    page_size = 16
    num_pages = 4

    key = torch.randn(
        num_tokens,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

    value = torch.randn_like(key)

    positions = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6],
        device=device,
        dtype=torch.int32,
    )

    slot_mapping = torch.tensor(
        [17, 2, -1, 31, 32, 47, 63],
        device=device,
        dtype=torch.int32,
    )

    half = rotary_dim // 2

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

    freqs = positions.float()[:, None] * inv_freq[None, :]

    cos_cache = freqs.cos().to(torch.float16)
    sin_cache = freqs.sin().to(torch.float16)

    max_positions = int(positions.max().item()) + 1

    full_cos_cache = torch.empty(
        max_positions,
        half,
        device=device,
        dtype=torch.float16,
    )

    full_sin_cache = torch.empty_like(full_cos_cache)

    full_cos_cache[positions.long()] = cos_cache
    full_sin_cache[positions.long()] = sin_cache

    k_cache = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

    v_cache = torch.randn_like(k_cache)

    expected_k = k_cache.clone()
    expected_v = v_cache.clone()

    valid = slot_mapping >= 0

    slots = slot_mapping[valid].long()

    pages = slots // page_size
    offsets = slots % page_size

    pos = positions[valid].long()

    cos = full_cos_cache[pos][:, None, :]
    sin = full_sin_cache[pos][:, None, :]

    rotated_key = apply_rope(
        key[valid],
        cos,
        sin,
        rotary_dim=rotary_dim,
    )

    expected_k[pages, offsets] = rotated_key
    expected_v[pages, offsets] = value[valid]

    fused_rope_fp16_append(
        key,
        value,
        positions,
        slot_mapping,
        full_cos_cache,
        full_sin_cache,
        k_cache,
        v_cache,
        rotary_dim=rotary_dim,
        page_size=page_size,
    )

    torch.cuda.synchronize()

    torch.testing.assert_close(
        k_cache,
        expected_k,
        rtol=2e-3,
        atol=2e-3,
    )

    assert torch.equal(v_cache, expected_v)
