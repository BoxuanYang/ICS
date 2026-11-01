# MiniSGLang KV Cache 分配

本文只讨论 `cache_type=naive`，说明 MiniSGLang 如何创建 KV physical pool、维护 page table，以及为新 token 分配物理 KV block。这里不考虑 radix cache 和 prefix cache 复用。

## 关键数据结构

| 数据结构 | 作用 |
|---|---|
| `MHAKVCache._kv_buffer` | 真正存放所有层的 K/V 数据 |
| `Engine.page_table` | 每个请求的逻辑 token 位置到物理 token slot 的映射 |
| `CacheManager.free_slots` | 尚未使用的物理 page 起始位置 |
| `TableManager._free_slots` | 尚未使用的 page-table 行号，即 request slot |
| `TableManager.token_pool` | 与 page table 同形状，保存每个请求的 token ID |
| `Req.cached_len/device_len` | 已写入 KV 的长度，以及本轮 attention 后应达到的长度 |

## 1. 服务启动时分配 KV physical pool

入口是 `python/minisgl/engine/engine.py` 中的 `Engine.__init__`。

模型权重加载完成后，`_determine_num_pages()` 根据剩余显存计算可分配的 page 数：

```text
每个 page 的字节数
= 2 × num_layers × local_kv_heads × head_dim × page_size × dtype_size

KV 可用显存
= memory_ratio × 模型加载前空闲显存 - 模型权重占用显存

num_pages = KV 可用显存 / 每个 page 的字节数
```

也可以通过 `num_page_override` 直接指定 page 数。TP 模式下，每个 rank 只保存本 rank 的 KV heads。

随后 `create_kvcache_pool()` 创建 `MHAKVCache`。底层一次性分配一个连续 CUDA tensor：

```python
self._kv_buffer = torch.empty(
    (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
    device=device,
    dtype=dtype,
)
```

第一维的 `2` 分别表示 K 和 V。Engine 实际传入 `num_pages + 1`，最后一个 page 专门给 CUDA Graph 的 dummy request 使用，不进入正常请求的空闲池。

启动时还会创建全局 page table：

```python
page_table.shape = (max_running_req + 1, aligned_max_seq_len)
page_table.dtype = torch.int32
page_table.device = CUDA
```

多出来的一行属于 dummy request。它的所有位置都指向额外 dummy page 的起始 token slot。

## 2. Page table 如何管理

全局 page table 虽然叫 page table，但其中存放的不是 page ID，而是展平后的物理 token slot：

```text
page_table[request_slot, logical_token_position] = physical_token_slot
```

例如 `page_size = 4`，某请求获得起始位置为 `8` 和 `20` 的两个物理 page，则对应表项是：

```text
[8, 9, 10, 11, 20, 21, 22, 23]
```

### 请求行管理

`TableManager` 管理 page-table 行：

1. 新请求通过 `TableManager.allocate()` 获得一个 `table_idx`。
2. `Req.table_idx` 在请求生命周期内保持不变。
3. 请求结束时，`TableManager.free(table_idx)` 将该行号放回空闲列表。

释放行号时不会清零旧 page-table 内容。后续请求只读取自身有效长度内的表项，而新分配会覆盖对应区间。

### Naive 模式

`NaivePrefixCache.match_prefix()` 永远返回 `cached_len = 0`，因此新请求不会复用其他请求的 KV，也不会向 page table 复制缓存前缀。每个请求都从逻辑位置 `0` 开始申请自己的物理 pages。

### Attention backend 使用

- FlashAttention/TRTLLM 从全局表中每隔 `page_size` 取一个起始 slot，再除以 `page_size` 得到真正的 page ID。
- FlashInfer 当前按 `page_size = 1` 使用，直接把各请求的物理 token slots 拼成 indices。

## 3. 给新 token 分配物理 KV block

每次执行 prefill 或 decode batch 前，`Scheduler._prepare_batch()` 都会先调用：

```python
self.cache_manager.allocate_paged(batch.reqs)
```

### 3.1 判断是否需要新 page

对每个请求计算：

```python
first_page = ceil(req.cached_len / page_size)
last_page  = ceil(req.device_len / page_size)
needed_pages = last_page - first_page
```

- Prefill 会一次分配覆盖 `[cached_len, device_len)` 所需的全部 pages。
- Decode 每次只有一个新 token；只有该 token 跨越 page 边界时才分配新 page。
- Naive 模式没有可淘汰的 prefix cache。调度器通过 `available_size` 在接收请求前检查容量；如果实际分配时仍没有足够空闲 pages，`NaivePrefixCache.evict()` 会直接报错。

### 3.2 更新的数据结构

分配发生时按以下顺序更新：

1. 从 `CacheManager.free_slots` 取出物理 page 起始位置，空闲列表缩短。
2. `_page_to_token()` 将每个 page 展开为连续的物理 token slots。
3. `_write_page_table()` 把这些 slots 写入对应 `table_idx` 的逻辑位置。
4. Scheduler 根据本轮逻辑 positions 查询 page table：

   ```python
   batch.out_loc = page_table[request_indices, token_positions]
   ```

5. 每个 attention layer 调用 `MHAKVCache.store_kv()`，按照 `batch.out_loc` 将本轮 K/V 写进 `_kv_buffer`。

因此，分配 page 只修改 allocator 和 page table；真正的 KV 内容是在模型逐层 forward 时写入。

### 3.3 Decode 单步状态变化

假设进入某个 decode step 时：

```text
cached_len = N
device_len = N + 1
```

本轮处理逻辑位置 `N` 的 token：

1. 必要时为位置 `N` 分配新 page，并更新 `page_table`。
2. 从 `page_table[table_idx, N]` 得到物理位置，写入该 token 的 K/V。
3. `Req.complete_one()` 执行：`cached_len = device_len`，然后 `device_len += 1`。
4. sampler 生成的下一个 token 写入 `token_pool[table_idx, N + 1]`，供下一轮 decode 使用。

新 token 分配过程中会修改：

- `CacheManager.free_slots`：跨页时减少一个空闲 page。
- `Engine.page_table`：写入新的逻辑到物理映射。
- `Batch.out_loc`：记录本轮 KV 的实际写入位置。
- `MHAKVCache._kv_buffer`：写入每一层的新 K/V。
- `Req.cached_len/device_len`：推进请求长度。
- `TableManager.token_pool`：保存 sampler 产生的下一个 token。

### 3.4 请求结束时释放

Naive 模式不会保留请求 KV。请求完成或被 abort 时：

1. `TableManager.free(req.table_idx)` 归还 request slot。
2. `CacheManager.cache_req(req, finished=True)` 取得该请求 page table 中 `[0, cached_len)` 的物理 slots。
3. `_free()` 每隔 `page_size` 取一个 page 起始位置，追加回 `CacheManager.free_slots`。

Page table 行本身不会清零，但已经失去所有权；该行下次分配给新请求时会被重新覆盖。

## 4. 多卡时如何切分 KV Cache

当前多卡使用 Tensor Parallel，KV Cache 按 **KV head** 切分，不按请求、token 或 page 切分。

每个 TP rank 都会创建自己的 `MHAKVCache`：

```text
[2, num_layers, num_pages + 1, page_size, local_kv_heads, head_dim]
```

其中：

```python
local_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
```

分为两种情况：

1. `num_kv_heads >= tp_size`

   要求 `num_kv_heads` 能被 `tp_size` 整除。不同 rank 保存不同的 KV heads。例如 8 个 KV heads、4 张卡时，每张卡保存 2 个 heads。

2. `num_kv_heads < tp_size`

   要求 `tp_size` 能被 `num_kv_heads` 整除。每个 rank 保存 1 个 KV head，同一个 head 会复制到多个 rank。例如 2 个 KV heads、8 张卡时，rank 0～3 保存 head 0，rank 4～7 保存 head 1。

K/V projection 权重使用同样的切分或复制规则，因此每个 rank 计算出的 K/V 正好写入自己的本地 KV Cache。

### Page allocator 是否跨卡切分

不会。每个 rank 都有独立但形状相同的：

- `MHAKVCache._kv_buffer`
- `Engine.page_table`
- `CacheManager.free_slots`

Rank 0 会把请求消息广播给其他 ranks，各 rank 的 Scheduler 按相同顺序执行相同的分配，所以 page table 和空闲 page 状态保持一致。例如所有 rank 都可能为某个 token 使用 physical slot `128`，但该 slot 位于各自 GPU 的本地 KV buffer 中，保存的是各自的 KV-head shard。

因此，多卡下的切分关系是：

```text
同一请求、同一逻辑 token、同一 page 映射
                         ×
不同 GPU 上的 KV-head shard
```

在固定 `num_pages` 下，KV heads 能正常切分时，每张卡的 KV Cache 字节数约为单卡的 `1 / tp_size`；自动计算容量时，更小的单页开销会允许服务创建更多 pages。当 KV heads 数少于卡数时，每卡至少保存 1 个 KV head，继续增加 TP 卡数不会再降低每卡单页的 KV 开销。

## 代码入口

- Physical pool：`python/minisgl/engine/engine.py`、`python/minisgl/kvcache/mha_pool.py`
- Page allocator：`python/minisgl/scheduler/cache.py`
- Request table：`python/minisgl/scheduler/table.py`
- 请求准入与 prefill：`python/minisgl/scheduler/prefill.py`
- Naive cache：`python/minisgl/kvcache/naive_cache.py`
- Batch 与 decode 状态推进：`python/minisgl/scheduler/scheduler.py`、`python/minisgl/core.py`
- 多卡 KV-head 切分：`python/minisgl/layers/linear.py`、`python/minisgl/models/weight.py`
- 多 rank 请求同步：`python/minisgl/scheduler/io.py`
- KV 实际写入：`python/minisgl/attention/fa.py`、`python/minisgl/attention/fi.py`
