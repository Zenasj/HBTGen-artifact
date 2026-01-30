import torch
shapes = [1, -12]
res1 = torch.broadcast_shapes(*shapes)
print(res1)
res2 = torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape