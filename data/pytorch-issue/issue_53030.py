import torch
import torch.nn as nn

torch.manual_seed(0)

a = torch.randn(16, 5)
b = torch.randn(16, 5)
labels = torch.empty(16).bernoulli_().mul_(2).sub_(1)

loss_fn = nn.CosineEmbeddingLoss()

print(loss_fn(a, b, labels))  # 0.4538
print(loss_fn(a, b, labels.unsqueeze(1)))  # 0.4810