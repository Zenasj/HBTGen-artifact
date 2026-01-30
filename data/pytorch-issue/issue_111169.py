import torch

x = img_to_tensor(im)
print(x.is_contiguous(memory_format=torch.channels_last))
print(x.is_contiguous(memory_format=torch.contiguous_format))