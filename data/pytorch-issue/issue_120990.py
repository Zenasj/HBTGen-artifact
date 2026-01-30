import torch

input_data = torch.tensor([1, 2, 3])
other_data = torch.tensor([4, 5, 6])
result = torch.dot(input_data, other_data)

print(result)

tensor(32)

import torch

input_data = torch.tensor([1, 2, 3])
other_data = torch.tensor([4, 5, 6])

input_data = input_data.cuda()
other_data = other_data.cuda()
result = torch.dot(input_data, other_data)

print(result)