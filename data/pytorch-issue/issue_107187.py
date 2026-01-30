import torch
import torch._inductor.config
torch._inductor.config.fallback_random = True

@torch.compile()
def foo():
    torch.manual_seed(3)
    return torch.rand([4])

print(foo())
print(foo())