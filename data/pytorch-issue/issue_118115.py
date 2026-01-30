import torch

x = torch.rand(2, 3, device="cpu").t()
x_mps = x.to(device="mps")
print(x)
print(x_mps)  # same

y = torch.rand(2, 3, device="cpu").t()
y_mps = y.to(device="mps")
print(y)
print(y_mps)  # same

z = torch.rand_like(x)
z_mps = z.to(device="mps")
print(z)
print(z_mps)  # same

print(x.addcdiv_(y, z))
print(x_mps.addcdiv_(y_mps, z_mps)) # NOT same!