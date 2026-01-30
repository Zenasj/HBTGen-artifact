import torch

a = torch.tensor(2854.)
b = torch.tensor(2855.)
c = torch.tensor(5877.)

ret = (a + 1) * c / b
print(ret)
# tensor(5877.0005) !!!

print(((a + 1) * c) > 2**24)
# tensor(True)