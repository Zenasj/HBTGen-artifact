py
import threading
import torch._dynamo

FOO = threading.local()
FOO.x = torch.zeros(1)

@torch.compile(backend='aot_eager')
def f(x):
    assert isinstance(FOO.x, torch.Tensor) 
    return torch.cat([x, FOO.x])
inp = torch.ones(1)
f(inp)