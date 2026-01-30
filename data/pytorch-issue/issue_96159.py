import torch
import torchaudio

supported_dtypes = [torch.float32]

def foo(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + torch.cos(x)

for dtype in supported_dtypes:
    print(f"Testing smoke_test_compile for {dtype}")
    x = torch.rand(3, 3, device="cuda").type(dtype)
    x_eager = foo(x)
    x_pt2 = torch.compile(foo)(x)
    print(torch.allclose(x_eager, x_pt2))

import torch

@torch.compile
def foo(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + torch.cos(x)

if __name__ == "__main__":
    import sys
    import ctypes
    sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
    print(foo(torch.rand(3, 3, device='cuda')))