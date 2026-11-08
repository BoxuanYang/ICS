"""Paged BF16 attention with a per-layer table of real KV addresses."""

import triton
import triton.language as tl
import torch


@triton.jit
def _store(
    K,
    V,
    Slots,
    KP,
    VP,
    KS: tl.constexpr,
    VS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    slot = tl.load(Slots + token)
    kp = tl.load(KP + slot).to(tl.pointer_type(tl.bfloat16))
    vp = tl.load(VP + slot).to(tl.pointer_type(tl.bfloat16))
    tl.store(kp + offset, tl.load(K + token * KS + offset, offset < WIDTH, 0), offset < WIDTH)
    tl.store(vp + offset, tl.load(V + token * VS + offset, offset < WIDTH, 0), offset < WIDTH)


def store_kv(k, v, slots, addresses, width):
    k, v = k.contiguous(), v.contiguous()
    _store[(k.shape[0],)](
        k,
        v,
        slots,
        addresses[0],
        addresses[1],
        k.stride(0),
        v.stride(0),
        width,
        triton.next_power_of_2(width),
    )


@triton.jit
def _gather_kv(K, V, Slots, KP, VP, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    slot = tl.load(Slots + token)
    kp = tl.load(KP + slot).to(tl.pointer_type(tl.bfloat16))
    vp = tl.load(VP + slot).to(tl.pointer_type(tl.bfloat16))
    tl.store(K + token * WIDTH + offsets, tl.load(kp + offsets, offsets < WIDTH, 0), offsets < WIDTH)
    tl.store(V + token * WIDTH + offsets, tl.load(vp + offsets, offsets < WIDTH, 0), offsets < WIDTH)


def gather_kv(slots, addresses, heads, dim):
    # Per-layer read-only attention workspace, not persistent KV allocation.
    shape = (slots.numel(), 1, heads, dim)
    k = torch.empty(shape, dtype=torch.bfloat16, device=slots.device)
    v = torch.empty_like(k)
    width = heads * dim
    _gather_kv[(slots.numel(),)](k, v, slots, addresses[0], addresses[1], width,
                               triton.next_power_of_2(width))
    return k, v


@triton.jit
def _gather_dense(K, V, KP, VP, Active, COUNT: tl.constexpr, WIDTH: tl.constexpr,
                  BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = offsets // WIDTH
    column = offsets % WIDTH
    valid = slot < COUNT
    enabled = tl.load(Active + slot, valid, 0)
    kp = tl.load(KP + slot, valid, 0).to(tl.pointer_type(tl.bfloat16))
    vp = tl.load(VP + slot, valid, 0).to(tl.pointer_type(tl.bfloat16))
    # All KV row addresses are at least 32-byte aligned. Rows remain contiguous
    # in groups of 16 BF16 elements even when adjacent rows belong to experts
    # at unrelated addresses. Make that invariant visible to vectorization.
    kp = tl.multiple_of(kp, 16)
    vp = tl.multiple_of(vp, 16)
    k = tl.load(tl.max_contiguous(kp + column, 16), valid & enabled, 0)
    v = tl.load(tl.max_contiguous(vp + column, 16), valid & enabled, 0)
    tl.store(K + offsets, k, valid)
    tl.store(V + offsets, v, valid)


def gather_dense_kv(k, v, addresses, active):
    count, _, heads, dim = k.shape
    width = heads * dim
    _gather_dense[(triton.cdiv(count * width, 2048),)](
        k, v, addresses[0], addresses[1], active, count, width, 2048)


@triton.jit
def _copy_slots(
    Addresses,
    Sources,
    Destinations,
    TOTAL_SLOTS: tl.constexpr,
    LAYERS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    layer = tl.program_id(1)
    kind = tl.program_id(2)
    offsets = tl.arange(0, BLOCK)
    source = tl.load(Sources + token)
    destination = tl.load(Destinations + token)
    base = (kind * LAYERS + layer) * TOTAL_SLOTS
    source_ptr = tl.load(Addresses + base + source).to(tl.pointer_type(tl.bfloat16))
    destination_ptr = tl.load(Addresses + base + destination).to(
        tl.pointer_type(tl.bfloat16)
    )
    values = tl.load(source_ptr + offsets, offsets < WIDTH)
    tl.store(destination_ptr + offsets, values, offsets < WIDTH)


def copy_slots(addresses, sources, destinations, width):
    """Copy complete multi-layer K/V blocks between logical slot addresses."""
    assert sources.numel() == destinations.numel()
    if sources.numel() == 0:
        return
    _, layers, total_slots = addresses.shape
    _copy_slots[(sources.numel(), layers, 2)](
        addresses,
        sources,
        destinations,
        total_slots,
        layers,
        width,
        triton.next_power_of_2(width),
    )


@triton.jit
def _decode_attention_split(
    Q,
    Table,
    KP,
    VP,
    SeqLens,
    PartialMax,
    PartialDenom,
    PartialAcc,
    QS: tl.constexpr,
    QH: tl.constexpr,
    TS: tl.constexpr,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    SPLITS: tl.constexpr,
):
    request = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    groups = tl.arange(0, GROUP_BLOCK)
    dims = tl.arange(0, D)
    heads = kv_head * GROUP + groups
    q = tl.load(
        Q + request * QS + heads[:, None] * QH + dims[None, :],
        groups[:, None] < GROUP,
        0.0,
    )
    klen = tl.load(SeqLens + request)
    maximum = tl.full((GROUP_BLOCK,), -float("inf"), tl.float32)
    denominator = tl.zeros((GROUP_BLOCK,), tl.float32)
    accumulator = tl.zeros((GROUP_BLOCK, D), tl.float32)
    block = split
    while block * BN < klen:
        columns = block * BN + tl.arange(0, BN)
        valid = columns < klen
        slots = tl.load(Table + request * TS + columns, valid, 0)
        kp = tl.load(KP + slots).to(tl.pointer_type(tl.bfloat16))
        vp = tl.load(VP + slots).to(tl.pointer_type(tl.bfloat16))
        k = tl.load(kp[None, :] + kv_head * D + dims[:, None], valid[None, :], 0.0)
        v = tl.load(vp[:, None] + kv_head * D + dims[None, :], valid[:, None], 0.0)
        # All query heads in one GQA group share this K/V head. Loading K and V
        # once here avoids the previous GROUP-fold redundant HBM traffic.
        score = tl.dot(q, k).to(tl.float32) * SCALE
        score = tl.where(valid[None, :], score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, axis=1))
        probability = tl.exp(score - new_max[:, None])
        correction = tl.exp(maximum - new_max)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probability.to(tl.bfloat16), v)
        denominator = denominator * correction + tl.sum(probability, axis=1)
        maximum = new_max
        block += SPLITS

    partial = (request * HEADS + heads) * SPLITS + split
    tl.store(PartialMax + partial, maximum, groups < GROUP)
    tl.store(PartialDenom + partial, denominator, groups < GROUP)
    tl.store(
        PartialAcc + partial[:, None] * D + dims[None, :],
        accumulator,
        groups[:, None] < GROUP,
    )


@triton.jit
def _decode_attention_reduce(
    O,
    SeqLens,
    PartialMax,
    PartialDenom,
    PartialAcc,
    OS: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, D)
    splits = tl.arange(0, SPLITS)
    base = (request * HEADS + head) * SPLITS
    maxima = tl.load(PartialMax + base + splits)
    maximum = tl.max(maxima, axis=0)
    correction = tl.exp(maxima - maximum)
    denominators = tl.load(PartialDenom + base + splits)
    denominator = tl.sum(denominators * correction, axis=0)
    accumulators = tl.load(
        PartialAcc + (base + splits[:, None]) * D + dims[None, :]
    )
    accumulator = tl.sum(accumulators * correction[:, None], axis=0)
    result = accumulator / denominator
    result = tl.where(tl.load(SeqLens + request) > 0, result, 0.0)
    tl.store(O + request * OS + head * D + dims, result)


def decode_attention(q, output, table, addresses, seq_lens, kv_heads):
    """CUDA-graph-safe batched decode over the fixed expert KV address table."""
    _, heads, dim = q.shape
    # A single program per head serializes a long sequence and leaves most SMs
    # idle when only a few requests own temporary KV. Eight interleaved splits
    # expose sequence parallelism; the second kernel combines their online
    # softmax states. PyTorch's CUDA-graph pool reuses these workspaces across
    # layers, so the split does not multiply persistent graph memory by depth.
    splits = 8
    group = heads // kv_heads
    partial_max = torch.empty(
        (q.shape[0], heads, splits), dtype=torch.float32, device=q.device
    )
    partial_denom = torch.empty_like(partial_max)
    partial_acc = torch.empty(
        (q.shape[0], heads, splits, dim), dtype=torch.float32, device=q.device
    )
    _decode_attention_split[(q.shape[0], kv_heads, splits)](
        q,
        table,
        addresses[0],
        addresses[1],
        seq_lens,
        partial_max,
        partial_denom,
        partial_acc,
        q.stride(0),
        q.stride(1),
        table.stride(0),
        heads,
        kv_heads,
        group,
        max(16, triton.next_power_of_2(group)),
        dim,
        dim**-0.5,
        32,
        splits,
    )
    _decode_attention_reduce[(q.shape[0], heads)](
        output,
        seq_lens,
        partial_max,
        partial_denom,
        partial_acc,
        output.stride(0),
        heads,
        dim,
        splits,
    )


@triton.jit
def _attention(
    Q,
    O,
    Table,
    KP,
    VP,
    QLEN,
    KLEN,
    PREFIX,
    QS: tl.constexpr,
    QH: tl.constexpr,
    OS: tl.constexpr,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    kv_head = head // (HEADS // KV_HEADS)
    dims = tl.arange(0, D)
    q = tl.load(Q + rows[:, None] * QS + head * QH + dims[None, :], rows[:, None] < QLEN, 0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, D), tl.float32)
    # Bottom-right causal mask: query row 0 is at position PREFIX.
    end = tl.minimum(KLEN, PREFIX + (tl.program_id(0) + 1) * BM)
    for start in range(0, tl.cdiv(end, BN)):
        columns = start * BN + tl.arange(0, BN)
        slots = tl.load(Table + columns, columns < KLEN, 0)
        kp = tl.load(KP + slots).to(tl.pointer_type(tl.bfloat16))
        vp = tl.load(VP + slots).to(tl.pointer_type(tl.bfloat16))
        k = tl.load(kp[None, :] + kv_head * D + dims[:, None], columns[None, :] < KLEN, 0)
        v = tl.load(vp[:, None] + kv_head * D + dims[None, :], columns[:, None] < KLEN, 0)
        score = tl.dot(q, k).to(tl.float32) * SCALE
        valid = (columns[None, :] < KLEN) & (columns[None, :] <= PREFIX + rows[:, None])
        score = tl.where(valid, score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        probability = tl.exp(score - new_max[:, None])
        correction = tl.exp(maximum - new_max)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probability.to(tl.bfloat16), v)
        denominator = denominator * correction + tl.sum(probability, 1)
        maximum = new_max
    result = accumulator / denominator[:, None]
    tl.store(O + rows[:, None] * OS + head * D + dims[None, :], result, rows[:, None] < QLEN)


def paged_attention(q, output, table, addresses, prefix, kv_heads):
    qlen, heads, dim = q.shape
    _attention[(triton.cdiv(qlen, 16), heads)](
        q,
        output,
        table,
        addresses[0],
        addresses[1],
        qlen,
        table.numel(),
        prefix,
        q.stride(0),
        q.stride(1),
        output.stride(0),
        heads,
        kv_heads,
        dim,
        dim**-0.5,
        16,
        32,
    )
