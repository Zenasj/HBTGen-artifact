import torch
x_cpu = torch.tensor([[1,2,3],[4,5,6]], device='cpu', dtype=torch.float32)
x_mps = x_cpu.detach().clone().to('mps')
print(f"{x_cpu.dtype = }")
print(f"{x_cpu.dtype == x_mps.dtype = }")
print(f"{x_cpu = }")
print(f"{(x_cpu == x_mps.cpu()).all() = }")
pinv_cpu = x_cpu.pinverse()
pinv_mps = x_mps.pinverse()
print(f"{pinv_cpu = }")
print(f"{((pinv_cpu - pinv_mps.cpu()).abs() > 1e-7).sum() = }")

# Ouput:
# x_cpu.dtype = torch.float32
# x_cpu.dtype == x_mps.dtype = True
# x_cpu = tensor([[1., 2., 3.],
#         [4., 5., 6.]])
# (x_cpu == x_mps.cpu()).all() = tensor(True)
# <stdin>:1: UserWarning: The operator 'aten::linalg_svd' is not currently supported on the MPS backend and will fall back to run on the CPU. This may have performance implications. (Triggered internally at /Users/hvaara/dev/pytorch/pytorch/aten/src/ATen/mps/MPSFallback.mm:13.)
#   pinv_mps = x_mps.pinverse()
# pinv_cpu = tensor([[-0.9444,  0.4444],
#         [-0.1111,  0.1111],
#         [ 0.7222, -0.2222]])
# ((pinv_cpu - pinv_mps.cpu()).abs() > 1e-7).sum() = tensor(0)