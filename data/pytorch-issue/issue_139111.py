import torch

@torch.compile(dynamic=True)
def f(a, b):
    a.add_(1)
    b.add_(1)
    return a

x = torch.ones(10)
f(x[:3], x[:3])

class HashableSymIntWrapper:
    def __init__(self, value: SymInt):
        self.value  = value

    def __hash__(self):
        try:
            return hash(self.value)
        except Exception:
            return hash(self.value.node.expr)