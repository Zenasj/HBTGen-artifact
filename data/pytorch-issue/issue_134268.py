import torch
import torch._dynamo
from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(backend="eager", fullgraph=True)
def fn(x):
    u49, u50 = x.tolist()
    torch._check_is_size(u49)
    torch._check_is_size(u50)
    torch._check((2*u49) % (u49 + u50) == 0)
    torch._check((2*u49)//(u49 + u50) != 0)
    if guard_size_oblivious((2*u49)//(u49 + u50) == 0):
        return torch.tensor(True)
    else:
        return torch.tensor(False)

fn(torch.tensor([3, 3]))