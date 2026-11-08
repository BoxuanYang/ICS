"""Private FlashInfer FA2 specialization for discontiguous expert-backed KV.

The regular FlashInfer module and its installed headers are never modified.
Only our uniquely named JIT module interprets page-stride 4 BF16 tensors as
64-bit address tables. K and V use different entry strides (page size is one)
so FlashInfer computes their independent offsets.
"""

from dataclasses import replace
from functools import cache
import hashlib
import os
from types import FunctionType

import torch


class IndirectFlashInferUnavailable(RuntimeError):
    """An older FlashInfer installation needs the dense compatibility path."""


@cache
def get_indirect_module(head_dim: int):
    import flashinfer.prefill as prefill
    from flashinfer.jit import env

    required = ("_gen_batch_prefill_primary_module",
                "_gen_batch_prefill_independent_paged_module")
    if any(not hasattr(prefill, name) for name in required):
        raise IndirectFlashInferUnavailable("FlashInfer lacks independent KV stride modules")

    original = env.FLASHINFER_INCLUDE_DIR / "flashinfer" / "page.cuh"
    source = original.read_text()
    for prefix, data in (("", "k_data"), ("v_", "v_data")):
        line = (f"    return page_idx * {prefix}stride_page + head_idx * "
                f"{prefix}stride_h + entry_idx * {prefix}stride_n + feat_idx;")
        if source.count(line) != 1:
            raise IndirectFlashInferUnavailable("Unsupported FlashInfer page.cuh for indirect KV")
        replacement = f"""
#ifdef __CUDA_ARCH__
    if ({prefix}stride_page == 4 && sizeof(DType) == 2) {{
      const auto address = __ldg(reinterpret_cast<const uint64_t*>({data}) + page_idx);
      const int64_t delta = int64_t(address) - int64_t(reinterpret_cast<uintptr_t>({data}));
      return size_t(delta / int64_t(sizeof(DType))) + head_idx * head_dim + feat_idx;
    }}
#endif
{line}"""
        source = source.replace(line, replacement)
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    suffix = "_minisgl_indirect_" + digest
    directory = env.FLASHINFER_GEN_SRC_DIR / suffix[1:]
    directory.mkdir(parents=True, exist_ok=True)
    header = directory / "page.cuh"
    if not header.exists() or header.read_text() != source:
        temporary = directory / f"page.{os.getpid()}.tmp"
        temporary.write_text(source)
        temporary.replace(header)

    def specialize(factory):
        def make(*args, **kwargs):
            spec = factory(*args, **kwargs)
            return replace(
                spec, name=spec.name + suffix,
                extra_cuda_cflags=[*(spec.extra_cuda_cflags or []),
                                   f"--pre-include={header}"],
                extra_include_dirs=[*(spec.extra_include_dirs or []), original.parent],
            )
        return make

    # Reuse FlashInfer's Python ABI adapter with private factories and op names.
    # This creates no global monkeypatch and cannot affect the pure-GPU path.
    original_factory = prefill.get_batch_prefill_module.__wrapped__
    namespace = dict(original_factory.__globals__)
    for name in ("_gen_batch_prefill_primary_module",
                 "_gen_batch_prefill_independent_paged_module"):
        namespace[name] = specialize(namespace[name])
    get_uri = namespace["get_batch_prefill_uri"]
    namespace["get_batch_prefill_uri"] = lambda *args: get_uri(*args) + suffix
    factory = FunctionType(original_factory.__code__, namespace)
    return factory("fa2", torch.bfloat16, torch.bfloat16, torch.bfloat16,
                   torch.int32, head_dim, head_dim, 0, False, False, False, False)


def make_pointer_views(addresses, heads: int, head_dim: int):
    """Own padded tables so fake tensor bounds are valid for every layer."""
    count = addresses.shape[-1]
    padding = (heads * head_dim + 3) // 4
    tables = torch.empty((*addresses.shape[:-1], count + padding),
                         dtype=torch.int64, device=addresses.device)
    tables[..., :count].copy_(addresses)
    tables[..., count:].zero_()
    views = []
    for layer in range(addresses.shape[1]):
        views.append(tuple(tables[kind, layer].view(torch.bfloat16).as_strided(
            (count, 1, heads, head_dim), (4, kind, head_dim, 1)) for kind in range(2)))
    return tables, views
