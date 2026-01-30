import torch

device = "mps"
diff = torch.tensor([[True, True], [True, True]], dtype=torch.bool)
diff = diff.T
target = torch.tensor([[0, 0], [0, 1]])

rcpu = torch.where(diff, target, 0)

diffmps = diff.to(device)
targetmps = target.to(device)

rmps = torch.where(diffmps, targetmps, 0)

print(rcpu)
print(rmps)