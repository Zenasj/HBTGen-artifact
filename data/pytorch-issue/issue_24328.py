import torch

torch.manual_seed(1234)

device = torch.device("cuda")

A = torch.randn(2, 3, 3, device=device)
B = torch.randn(2, 3, 5, device=device)

# CPU version works fine
solution_cpu, _ = torch.solve(B.cpu(), A.cpu())
print("CPU: ", solution_cpu)

solution_gpu, _ = torch.solve(B, A)
# The GPU solution seems to exist on the device
print("Solution Device = ", solution_gpu.device)
# Using the solution crashes
solution_gpu * 1
print("GPU: ", solution_gpu)