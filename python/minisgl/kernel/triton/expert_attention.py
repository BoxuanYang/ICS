"""Paged BF16 attention with a per-layer table of real KV addresses."""

import triton
import triton.language as tl


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
