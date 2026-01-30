import torch

@torch.jit.overload  # noqa: F811
def identity(x1):  # noqa: F811
    # type: (str) -> str
    pass

@torch.jit.overload  # noqa: F811
def identity(x1=1.0):  # noqa: F811
    # type: (float) -> float
    pass

def identity(x1=1.0):  # noqa: F811
    return x1

print(torch.jit.script(identity))