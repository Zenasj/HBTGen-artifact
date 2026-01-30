import torch

v_cpu = torch.tensor([[1.0, 2.0, 3.0]])
v_mps = torch.tensor([[1.0, 2.0, 3.0]], device="mps")
norm_cpu = v_cpu.norm(2, dim=[])
norm_mps = v_mps.norm(2, dim=[])
print(f"v_mps.norm(2, dim=[]) = {norm_mps} should be {norm_cpu}")

v_cpu = torch.tensor([[1.0, 2.0, 3.0]])
v_mps = torch.tensor([[1.0, 2.0, 3.0]], device="mps")
norm_cpu = v_cpu.norm(2, dim=0)
norm_mps = v_mps.norm(2, dim=0)
print(f"v_mps.norm(2, dim=0) = {norm_mps} should be {norm_cpu}")

# prints:
# v_mps.norm(2) = 3.741657257080078 should be 3.7416574954986572
# v_mps.norm(2, 0) = tensor([3.7417, 2.0000, 3.0000], device='mps:0') should be tensor([1., 2., 3.])

import torch

v_cpu = torch.tensor([[1.0, 2.0, 3.0]])
v_mps = torch.tensor([[1.0, 2.0, 3.0]], device="mps")
norm_cpu = v_cpu.norm(2, dim=0)
norm_mps = v_mps.norm(2, dim=0)
print(f"v_mps.norm(2, dim=0) = {norm_mps} should be {norm_cpu}")

v_cpu = torch.tensor([[1.0, 2.0, 3.0]])
v_mps = torch.tensor([[1.0, 2.0, 3.0]], device="mps")
norm_cpu = v_cpu.norm(2, dim=[])
norm_mps = v_mps.norm(2, dim=[])
print(f"v_mps.norm(2, dim=[]) = {norm_mps} should be {norm_cpu}")

# v_mps.norm(2, dim=0) = tensor([1., 2., 3.], device='mps:0') should be tensor([1., 2., 3.])
# /AppleInternal/Library/BuildRoots/c651a45f-806e-11ed-a221-7ef33c48bc85/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPSNDArray/Kernels/MPSNDArrayMultiaryKernel.mm:1596: failed assertion `Error: Invalid KernelDAG, equalShape for destination failed'
# zsh: abort      python ./norm_crash.py