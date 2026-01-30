import torch
myMatrix = torch.tensor((4, 4), dtype=torch.cfloat, device='mps')
print(myMatrix)
# tensor([4.+0.j, 4.+0.j], device='mps:0')