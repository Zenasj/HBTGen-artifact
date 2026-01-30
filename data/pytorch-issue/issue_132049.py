import torch
input = torch.tensor([0.5, 2.5, 2.0], dtype=torch.float32)
target = torch.tensor([0.0, 2.0, 1.5], dtype=torch.float32)
weights = torch.tensor([1.0, 0.5, 1.5], dtype=torch.float32)

loss = weighted_huber_loss(input, target, weights, delta=1.0)
print(loss)