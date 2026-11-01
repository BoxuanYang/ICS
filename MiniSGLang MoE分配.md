# ICS MoE 专家显存分配与权重加载

ICS 中相关代码主要在以下三个文件：

- `python/minisgl/layers/moe.py`：定义专家权重的形状。
- `python/minisgl/models/weight.py`：读取、切分并组合权重。
- `python/minisgl/engine/engine.py`：创建模型并触发权重加载。

## 1. 专家权重定义

MoE 逻辑上有 `gate_proj`、`up_proj` 和 `down_proj` 三组参数。ICS 将
`gate_proj` 与 `up_proj` 融合为 `gate_up_proj`，因此 `MoELayer` 运行时只保存
下面两个专家权重 tensor：

```python
self.gate_up_proj = torch.empty(
    num_experts,
    2 * intermediate_size_per_partition,
    hidden_size,
)

self.down_proj = torch.empty(
    num_experts,
    hidden_size,
    intermediate_size_per_partition,
)
```

模型是在 `meta device` 上创建的，所以这里暂时只记录形状，并不真正分配显存。

## 2. 权重加载

`python/minisgl/engine/engine.py` 先创建模型，再调用权重加载：

```python
with torch.device("meta"), torch_dtype(config.dtype):
    self.model = create_model(config.model_config)

self.model.load_state_dict(self._load_weight_state_dict(config))
```

实际读取逻辑位于 `python/minisgl/models/weight.py`：

```python
with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
    raw = f.get_tensor(name)
    tensor = _shard_tensor(name, raw, tp_rank, tp_size, num_kv_heads)
```

权重被直接读取到当前 GPU，然后按 Tensor Parallel rank 切分：

- `gate_proj`、`up_proj` 按 `dim=0` 切分。
- `down_proj` 按 `dim=1` 切分。
- 同一专家的 `gate_proj` 和 `up_proj` 使用 `torch.cat` 合并。
- 所有专家使用 `torch.stack` 打包成 `gate_up_proj` 和 `down_proj`。

最后，`load_state_dict` 用这些 CUDA tensor 替换模型中的 meta tensor，此时专家权重正式常驻 VRAM。

## 3. 多卡场景下的 MoE 权重切分

当前实现使用 Tensor Parallel，不是 Expert Parallel。每个 TP rank 都保存全部 `num_experts`，只切分每个专家的 intermediate dimension。

要求 `moe_intermediate_size` 可以被 `tp_size` 整除：

```python
# python/minisgl/layers/moe.py
intermediate_size_per_partition = div_even(intermediate_size, tp_size)
```

### 3.1 Checkpoint 权重如何切分

单个专家在 checkpoint 中的权重形状为：

```text
gate_proj: [moe_intermediate_size, hidden_size]
up_proj:   [moe_intermediate_size, hidden_size]
down_proj: [hidden_size, moe_intermediate_size]
```

`python/minisgl/models/weight.py` 定义了切分维度：

```python
_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]
```

实际切分代码是：

```python
if any(key.count(sub) for sub in _SPLIT_DIM_0):
    return value.chunk(n, dim=0)[r].clone()
elif any(key.count(sub) for sub in _SPLIT_DIM_1):
    return value.chunk(n, dim=1)[r].clone()
```

因此 rank `r` 获得：

```text
gate_proj_r: [moe_intermediate_size / tp_size, hidden_size]   # 按 dim=0 切分
up_proj_r:   [moe_intermediate_size / tp_size, hidden_size]   # 按 dim=0 切分
down_proj_r: [hidden_size, moe_intermediate_size / tp_size]   # 按 dim=1 切分
```

随后 loader 先将本 rank 的 `gate_proj_r` 和 `up_proj_r` 合并：

```python
gate_up_proj_r = torch.cat([gate_proj_r, up_proj_r], dim=0)
```

再把全部 expert ID 沿第 0 维打包：

```python
torch.stack(experts, dim=0)
```

所以每个 rank 最终保存：

```text
gate_up_proj: [num_experts, 2 * moe_intermediate_size / tp_size, hidden_size]
down_proj:    [num_experts, hidden_size, moe_intermediate_size / tp_size]
```

例如 `num_experts=64`、`moe_intermediate_size=8192`、`tp_size=4` 时，每张卡仍有 64 个专家，但每个专家只保存 `2048` 宽度的 intermediate shard。

### 3.2 Router 是否切分

Router 不切分。`MoEMLP` 使用 `LinearReplicated` 创建 gate：

```python
self.gate = LinearReplicated(
    config.hidden_size,
    config.num_experts,
    has_bias=False,
)
```

因此每个 rank 都保存完整的 `[num_experts, hidden_size]` router 权重，并计算相同的 router logits 和 top-k expert IDs。

### 3.3 Forward 如何合并各卡结果

每个 rank 使用相同的 token 和 expert IDs，但只使用本地 `moe_intermediate_size / tp_size` 宽度的权重 shard：

```text
hidden [hidden_size]
  -> local gate/up [moe_intermediate_size / tp_size]
  -> activation
  -> local down [hidden_size]
  -> 本 rank 的 partial output [hidden_size]
```

`python/minisgl/layers/moe.py` 最后对 partial output 做 all-reduce sum：

```python
final_hidden_states = ctx.moe_backend.forward(
    hidden_states=hidden_states,
    w1=self.gate_up_proj,
    w2=self.down_proj,
    gating_output=router_logits,
    ...
)

if self.tp_size > 1:
    final_hidden_states = self._comm.all_reduce(final_hidden_states)
```

这里不需要在各卡之间传输完整 intermediate tensor，只需要对最终 `[num_tokens, hidden_size]` 输出求和。

### 3.4 每卡专家权重大小

固定模型配置下，每层每个 rank 的专家权重元素数为：

```text
num_experts × (
    2 × moe_intermediate_size / tp_size × hidden_size
    + hidden_size × moe_intermediate_size / tp_size
)
= 3 × num_experts × hidden_size × moe_intermediate_size / tp_size
```

因此专家主体权重随 `tp_size` 近似按 `1 / tp_size` 缩小；但 router `[num_experts, hidden_size]` 在所有 rank 上完整复制。

需要注意，当前 `load_weight()` 会先通过 `safetensors` 把原始 tensor 读到当前 GPU，再取本 rank 的 shard，所以加载阶段的峰值显存不一定只有最终 shard 大小。

## 4. 当前结论

目前每个 TP rank 都会加载全部专家，只切分每个专家的中间维度；代码中还没有 cold expert offload、CPU expert cache 或按需加载机制。
