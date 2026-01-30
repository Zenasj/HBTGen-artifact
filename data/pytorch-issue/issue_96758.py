import typing as t
import torch

def foo(x: t.Dict):
    return x

scripted_fn = torch.jit.script(foo)