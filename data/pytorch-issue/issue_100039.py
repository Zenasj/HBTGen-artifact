import torch

x = 0.0
y = torch.tensor(0.0)
eps = torch.tensor(0.001)
for i in range(1_000_000):
    x += eps.item()
    y += eps

print(f"x = {x}")
print(f"y = {y.item()}")

# x = 1000.0000474974513
# y = 991.1415405273438

import torch
torch.set_printoptions(precision=19)

eps = torch.tensor(0.001)

print(eps)
print(eps.item())

# tensor(0.0010000000474974513)
# 0.0010000000474974513

x = 0.0
y = torch.tensor(0.0)
eps = torch.tensor(0.001)
for i in range(2):
    x += eps.item()
    y += eps

print(f"x = {x}")
print(f"y = {y.item()}")

# x = 0.0020000000949949026
# y = 0.0020000000949949026

x = 0.0
y = torch.tensor(0.0)
eps = torch.tensor(0.001)
for i in range(3):
    x += eps.item()
    y += eps

print(f"x = {x}")
print(f"y = {y.item()}")

# x = 0.003000000142492354
# y = 0.003000000026077032