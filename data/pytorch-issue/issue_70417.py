import torch
import torch.nn as nn

torch.manual_seed(seed=0)

target = torch.zeros(2, 2).long().to("cuda:1")   

target[0][0] = 3
target[1][0] = 5

logits = torch.randn(2, 10).to("cuda:1")

print(nn.functional.cross_entropy(logits.cpu(), target[:, 0].cpu(), reduction='none'))
print(nn.functional.cross_entropy(logits, target[:, 0], reduction='none'))

tensor([3.0468, 4.3090])
tensor([3.0468, 2.0217], device='cuda:1')

tensor([3.0468, 4.3090])
tensor([3.0468, 4.3090], device='cuda:0')