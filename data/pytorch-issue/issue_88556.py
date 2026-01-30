import torch

input_data = torch.arange(0, 50, 0.5).cuda()
output_data = torch.cumprod(input_data, dim=0)
print(input_data)
print(output_data)