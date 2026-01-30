import torch.nn as nn

import torch
import torch.nn.functional as F
from torch import Tensor

from typing import Any, Callable

torch.set_default_device("cuda")

weight = torch.randn(50304, 768, dtype=torch.bfloat16, requires_grad=True)
inputs = torch.randint(0, 50304, (65536,), dtype=torch.int32)
grads = torch.randn(65536, 768, dtype=torch.bfloat16)


def bench(
    f: Callable[..., Any],  # pyright: ignore[reportExplicitAny]
    name: str | None = None,
    warmup: int = 10,
    display: bool = True,
    profile: bool = False,
):
    from triton.testing import do_bench  # pyright: ignore[reportMissingTypeStubs,reportUnknownVariableType]

    for _ in range(warmup):
        f()

    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")

    ms_per_iter: float = do_bench(lambda: f(), rep=100)  # pyright: ignore[reportAssignmentType,reportAny]
    if name is None:
        res = ms_per_iter
    else:
        res = f"{name}: {ms_per_iter:.3f}ms"
    if display:
        print(res)
    return res


@torch.compile  # pyright: ignore[reportUnknownMemberType]
def f(inputs: Tensor, weight: Tensor):
    return F.embedding(inputs, weight).sum()


@torch.compile  # pyright: ignore[reportUnknownMemberType]
def g1(inputs: Tensor, weight: Tensor, grads: Tensor) -> Tensor:
    def fwd(w: Tensor):
        return F.embedding(inputs, w)

    (_, vjpfunc) = torch.func.vjp(fwd, weight)  # pyright: ignore[reportAssignmentType,reportUnknownMemberType,reportPrivateImportUsage,reportAny,reportUnknownVariableType]
    return vjpfunc(grads)[0].sum()  # pyright: ignore[reportAny]


@torch.compile  # pyright: ignore[reportUnknownMemberType]
def g2(inputs: Tensor, weight: Tensor, grads: Tensor):
    _ = F.embedding(inputs, weight).backward(grads)  # pyright: ignore[reportUnknownMemberType]


@torch.compile  # pyright: ignore[reportUnknownMemberType]
def g3(inputs: Tensor, weight: Tensor, grads: Tensor) -> Tensor:
    def fwd(w: Tensor):
        return w.index_select(0, inputs)

    (_, vjpfunc) = torch.func.vjp(fwd, weight)  # pyright: ignore[reportAssignmentType,reportUnknownMemberType,reportPrivateImportUsage,reportAny,reportUnknownVariableType]
    return vjpfunc(grads)[0].sum()  # pyright: ignore[reportAny]


_ = bench(lambda: f(inputs, weight))
_ = bench(lambda: f(inputs, weight))
_ = bench(lambda: f(inputs, weight))
_ = bench(lambda: g1(inputs, weight, grads))
_ = bench(lambda: g2(inputs, weight, grads))
_ = bench(lambda: g3(inputs, weight, grads))