import torch

bins = 10
max = 8
min = 176
input_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int8)

print(torch.histc(input_tensor.cuda(), bins=bins, min=min, max=max)) 
# tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 5], device='cuda:0', dtype=torch.int8)
print(torch.histc(input_tensor, bins=bins, min=min, max=max)) 
# RuntimeError: torch.histc: max must be larger than min