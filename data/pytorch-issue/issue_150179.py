import torch
input_size = (2, 3)
input_stride = (1, 2)
dtype = torch.float32
layout = torch.strided
device = torch.device('cuda:0')
requires_grad = False
pin_memory = True
result = torch.empty_strided(input_size, input_stride, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, pin_memory=pin_memory)