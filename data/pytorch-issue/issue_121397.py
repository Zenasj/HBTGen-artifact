import torch

input_data = torch.randn(3, 3)
condition = torch.rand(3, 3) > 0.5
other_data = torch.ones(3, 3)
output = torch.tensor(0)

output = torch.where(condition, input_data, other_data, out=output)

import torch

torch.set_default_device("mps")
input_data = torch.randn(3, 3)
condition = torch.rand(3, 3) > 0.5
other_data = torch.ones(3, 3)
output = torch.tensor(0)

print(torch.where(condition, input_data, other_data, out=output))