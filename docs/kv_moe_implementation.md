# KV–MoE 实现报告

当前范围：单卡、Qwen3-30B-A3B、BF16、`page_size=1`。以本次澄清为准，移除了旧方案中的 DRAINING、按压力下降恢复和 F_max 门限。

## 改动

| 部分 | 实现 |
| --- | --- |
| `moe/offload.py` | 读取 profiling 的 960 个三元组，只允许驱逐名单内专家，按频率升序选择；保留 `check_stop_moe_eviction()`。 |
| `moe/storage.py`、`kvcache/unified.py` | 每个专家的 gate/up、down 两段原始 storage 各对应一个 KV region；固定 pool 优先，尾部不足一个完整 unit 的字节不分配。 |
| `moe/memory.py` | 驱逐时先切 CPU 路由，再开放 buffer；两个 region 的 live KV 都归零时立即 fetch-back 完整权重，随后恢复 GPU 路由。 |
| `moe/kt.py`、GPU MoE kernel | 启动时用 KT 加载全部 CPU 专家；原地更新 KT expert mask；GPU 跳过 CPU 专家，CPU/GPU 并行计算后相加。Router 和 routing weight 保持原逻辑。 |
| `kvcache/expert_pool.py`、`attention/expert.py`、`kernel/triton/expert_attention.py` | 专家 storage 直接存放 KV；Triton attention 通过真实地址表读写，无临时 KV 到固定 pool 的复制。 |
| Engine、Scheduler、CLI | 接通分配、请求结束释放、prefix-cache 生命周期和 eager restoration；增加 `--expert-profile`、`--kt-cpu-threads`。 |
| `third_party/ktransformers` | 恢复递归子模块，保留主仓库固定的 `a02efe36b5802e7ca79a6b29146e01cf3b97d305`。 |

## 地址和执行顺序

请求 page table 存全局 slot ID，地址表把它映射到每层 K/V 的实际 GPU 指针。一个 unit 包含一个 token 在全部层的 K/V；同一请求的不同 page 可以来自固定 pool 或任意专家的 buffer，无需连续。

此模式支持 decode CUDA graph，但仍关闭跨 batch overlap，在已完成的 batch 之间改变内存所有权。graph 始终捕获 KT 的混合 CPU/GPU 路径，并在 replay 时读取固定地址的 CPU/GPU expert mask；pointer-table attention 使用固定地址的 page-table、seq-len capture buffer，因此驱逐和恢复不需要重新 capture。单次 MoE 内仍通过 KT 的 stream callback 重叠 CPU/GPU 计算。恢复不等全局压力下降，也不排序：某个专家两段 buffer 的最后一个 KV 引用释放即恢复。Radix 缓存保留的引用仍计为 live；demo 使用 naive cache，使请求结束即可观察恢复。

KT 的 CPU 内部权重可能经过重排，因此另存候选专家的原始 BF16 CPU 副本用于逐字节恢复。启动仍要求显存能容纳整个 BF16 模型；本功能扩展的是运行中的 KV 容量。

## 运行

按 README 安装 Mini-SGLang 和固定版本的 KT（Linux、CUDA toolkit、PyTorch 2.9.1）。拉取依赖：

```bash
git submodule update --init --recursive
# 以后更新主仓库及已存在的子模块：
git pull --recurse-submodules
```

启动和发送两轮并发请求：

```bash
bash scripts/demo_kv_moe.sh /path/to/Qwen3-30B-A3B
# 另一终端：
python scripts/check_kv_moe.py
```

Demo 固定 pool 仅 128 tokens，长 prompt 触发驱逐。服务日志应出现 `KV-MoE evict` 和对应的 `KV-MoE restore`；第二轮验证恢复后可再次驱逐。HTTP 脚本校验请求成功和非空文本，权重与数值正确性由下列测试验证。

```bash
PYTHONPATH=python python -m pytest -o addopts='' \
  tests/core/test_expert_offload.py tests/core/test_expert_storage.py \
  tests/core/test_unified_kv_allocator.py tests/core/test_expert_memory_manager.py \
  tests/core/test_expert_cache.py tests/kernel/test_expert_kv.py \
  tests/kernel/test_kt_experts.py tests/integration/test_kv_moe_demo.py
```

测试覆盖候选限制、驱逐顺序、重复/过期释放、eager restoration、调度器两次分配释放、非连续 KV attention、尾部保护、GPU 跳过已覆盖权重，以及真实 KT 混合/全 CPU 专家与全 GPU 输出比较。CUDA/KT 缺失时相关测试会跳过，跳过不代表通过。

## 验证结果

本机 WSL2、RTX 4060 Laptop 8GB、CUDA 12.8、PyTorch 2.9.1、固定版本 KT AVX2 BF16：**32 passed，0 skipped**。Ruff 和 `git diff --check` 通过。

端到端测试使用小型真实 Qwen3 MoE checkpoint 和实际 HTTP 服务，两轮各两个并发请求，覆盖 chunked prefill、decode、驱逐和立即恢复；四次响应成功且 greedy 输出一致。CPU 专家计算使用编译后的 KT 原生 kernel，没有替代实现。

完整 Qwen3-30B-A3B 因本机显存不足未实机运行；AMX 路径也未在本机验证。提供的 30B 启动脚本仍需在容量足够的目标服务器验收。

实现没有加入多卡、量化、235B、动态 SLO 策略或额外服务接口。当前 attention 是便于核对正确性的 Triton 实现，吞吐和 TTFT/TBT 收益需要在目标服务器上测量。
