import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class A:
    a: int


def fn() -> None:
    a = A(1)


comp_out = torch._dynamo.optimize(nopython=True)(fn)()