import torch
shape = (2,8,1,2)
i=torch.randint(1, shape, device='cuda').contiguous(memory_format=torch.channels_last)
x=torch.randn(shape, requires_grad=True, device='cuda')
x[i].sum().backward()