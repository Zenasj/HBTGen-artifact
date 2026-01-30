import torch

mps_device = torch.device("mps")

x = 0.1

cpu_tensor = torch.exp(torch.tensor(x))
mps_tensor = torch.exp(torch.tensor(x, device=mps_device))

print(cpu_tensor - cpu_tensor) # prints 0
print(mps_tensor - mps_tensor) # prints 0
print(cpu_tensor - mps_tensor) # prints 1.1921e-07
print(cpu_tensor - mps_tensor.cpu()) # prints 1.1921e-07
print(cpu_tensor.to(mps_device) - mps_tensor) # prints 1.1921e-07