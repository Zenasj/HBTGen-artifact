python
import torch
import torch.nn as nn

batch_size = 2
time_steps = 3
embedding_dim = 4

inputx = torch.randn(batch_size, time_steps, embedding_dim)  # N * L * C

linear = nn.Linear(embedding_dim, 3, bias=False)
wn_linear = torch.nn.utils.weight_norm(linear)
wn_linear_output = wn_linear(inputx)

weight_direction = linear.weight / linear.weight.norm(dim=1, keepdim=True)

print(linear.weight)
print(weight_direction)
print(wn_linear.weight_v)