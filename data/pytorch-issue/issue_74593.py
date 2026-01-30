import torch

output = upsample(h, output_size=input.size())
# input.size() is torch.Size([1, 16, 12, 12])
print(output.size())  # torch.Size([1, 16, 12, 12])

output = upsample(h)
print(output.size())  # torch.Size([1, 16, 11, 11])