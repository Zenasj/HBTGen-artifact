import torch
import triton

@torch.compile
def foo(x: torch.Tensor) -> torch.Tensor:
  return torch.sin(x) + torch.cos(x)

x=torch.rand(3, 3, device="cuda")
print(foo(x))
# And check that CUDA versions match
cuda_version = torch.version.cuda
ptxas_version = triton.backends.nvidia.compiler.get_ptxas_version().decode("ascii")
assert cuda_version in ptxas_version, f"CUDA version mismatch: torch build with {cuda_version}, but Triton uses ptxs {ptxas_version}"