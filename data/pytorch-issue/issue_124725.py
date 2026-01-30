py
import torch
from dataclasses import dataclass
from dataclasses import astuple, dataclass
import torch.utils
import torch.utils.checkpoint
import lightning as L

@dataclass
class D:
    s: torch.tensor
    y: torch.tensor
    # easy unpacking
    def __iter__(self):
        return iter(astuple(self))


def foo_dataclass(x:torch.Tensor):
    return D(s=x.mean(), y=torch.ones_like(x))

def foo_tuple(x:torch.Tensor):
    return (x.mean(), torch.ones_like(x))
a = torch.randn(40)
a.requires_grad=True

print("Dataclass:")
r = torch.utils.checkpoint.checkpoint(foo_dataclass, a, use_reentrant=True)
print("Checkpointed: ", r.s.requires_grad)
r2 = foo_dataclass(a)
print("Non-Checkpointed: ", r2.s.requires_grad)

a = torch.randn(40)
a.requires_grad=True
print("Tuple:")
r = torch.utils.checkpoint.checkpoint(foo_tuple, a, use_reentrant=True)
print("Checkpointed: ", r[0].requires_grad)
r2 = foo_tuple(a)
print("Non-Checkpointed: ", r2[0].requires_grad)