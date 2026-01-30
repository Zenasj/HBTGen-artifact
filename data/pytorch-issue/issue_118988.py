py
import torch

class T:
    def method(self):
        return self

t = T()
f = t.method

@torch.compile(backend="eager", fullgraph=True)
def fn():
    return f()

fn()