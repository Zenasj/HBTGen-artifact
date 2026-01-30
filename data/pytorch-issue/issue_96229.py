import torch

@torch.compile
def foo(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + torch.cos(x)

if __name__ == "__main__":
    import sys
    import ctypes
    sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
    print(foo(torch.rand(3, 3, device='cuda')))