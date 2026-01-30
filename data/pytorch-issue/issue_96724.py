import torch

torch.use_deterministic_algorithms(True)

x = torch.zeros(5, 3).cuda()
x[[1, 3], :] = 2
print(x)