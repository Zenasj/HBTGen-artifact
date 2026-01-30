import torch

before = torch.randperm(134217728)
after = before.to('cuda:0')
print(torch.max(before))  # tensor(134217727)
print(torch.max(after))  # tensor(1476362372, device='cuda:0') not always the same but never 134217727