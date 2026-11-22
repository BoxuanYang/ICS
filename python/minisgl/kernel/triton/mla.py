"""BF16 MLA rows in discontiguous expert storage."""

import torch
import triton
import triton.language as tl


@triton.jit
def _transfer(
    Rows, Slots, Addresses, WIDTH: tl.constexpr, BLOCK: tl.constexpr, STORE: tl.constexpr
):
    i = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    slot = tl.load(Slots + i)
    addr = tl.load(Addresses + slot).to(tl.pointer_type(tl.bfloat16))
    row = Rows + i * WIDTH
    if STORE:
        tl.store(addr + col, tl.load(row + col, col < WIDTH, 0), col < WIDTH)
    else:
        tl.store(row + col, tl.load(addr + col, col < WIDTH, 0), col < WIDTH)


def transfer_rows(rows, slots, addresses, store):
    _transfer[(len(slots),)](
        rows, slots, addresses, rows.shape[-1], triton.next_power_of_2(rows.shape[-1]), store
    )


@triton.jit
def _decode(
    Q,
    PE,
    Out,
    Addresses,
    Table,
    Req,
    Length,
    TABLE_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    H: tl.constexpr,
    SCALE: tl.constexpr,
):
    i = tl.program_id(0)
    h = tl.arange(0, H)
    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    t = tl.arange(0, 16)
    q = tl.load(Q + (i * HEADS + h[:, None]) * 512 + d[None, :], h[:, None] < HEADS, 0)
    p = tl.load(PE + (i * HEADS + h[:, None]) * 64 + r[None, :], h[:, None] < HEADS, 0)
    length = tl.load(Length + i)
    table = Table + tl.load(Req + i) * TABLE_STRIDE
    acc = tl.full((H, 512), 0, tl.float32)
    denom = tl.full((H,), 0, tl.float32)
    maximum = tl.full((H,), -float("inf"), tl.float32)
    for block in range(tl.cdiv(length, 16)):
        pos = block * 16 + t
        slots = tl.load(table + pos, pos < length, 0)
        ptr = tl.load(Addresses + slots).to(tl.pointer_type(tl.bfloat16))
        c = tl.load(ptr[:, None] + d[None, :], pos[:, None] < length, 0)
        k = tl.load(ptr[:, None] + 512 + r[None, :], pos[:, None] < length, 0)
        logits = (tl.dot(q, tl.trans(c)) + tl.dot(p, tl.trans(k))) * SCALE
        logits = tl.where(pos[None, :] < length, logits, -float("inf"))
        m = tl.maximum(maximum, tl.max(logits, 1))
        weights = tl.exp(logits - m[:, None])
        alpha = tl.exp(maximum - m)
        acc = acc * alpha[:, None] + tl.dot(weights.to(c.dtype), c)
        denom = denom * alpha + tl.sum(weights, 1)
        maximum = m
    tl.store(
        Out + (i * HEADS + h[:, None]) * 512 + d[None, :], acc / denom[:, None], h[:, None] < HEADS
    )


def indirect_decode(q, pe, addresses, table, req, length, scale):
    out = torch.empty_like(q)
    _decode[(len(q),)](
        q,
        pe,
        out,
        addresses,
        table,
        req,
        length,
        table.stride(0),
        q.shape[1],
        max(16, triton.next_power_of_2(q.shape[1])),
        scale,
    )
    return out
