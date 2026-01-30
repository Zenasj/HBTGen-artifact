import torch
device = torch.device("mps")
x = torch.tensor([],device=device)
result = x.topk(0)
print(result)