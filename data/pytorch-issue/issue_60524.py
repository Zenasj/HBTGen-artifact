import torch
from torch import Tensor
from torch.nn import AdaptiveAvgPool3d

module: AdaptiveAvgPool3d = AdaptiveAvgPool3d(output_size=2)
inputs: Tensor = torch.rand(size=(1, 1, 2, 2, 4))
inputs.requires_grad = True
output_cpu: Tensor = module(inputs)
mat: Tensor = torch.rand_like(output_cpu)
derivative_cpu_torch: Tensor = torch.autograd.grad(output_cpu, inputs, mat)[0]

module.to("cuda")
inputs = inputs.to(device="cuda")
mat = mat.to(device="cuda")
output_cuda: Tensor = module(inputs)
derivative_cuda_torch: Tensor = torch.autograd.grad(output_cuda, inputs, mat)[0]

print("Outputs identical?", torch.allclose(output_cpu, output_cuda.to(device="cpu")))
print(
    "Torch derivatives (cuda+cpu) match?",
    torch.allclose(derivative_cpu_torch, derivative_cuda_torch.to(device="cpu")),
)
print("cpu derivative torch:", derivative_cpu_torch)
print("cuda derivative torch:", derivative_cuda_torch)