import torch.nn as nn

import torch
torch.__version__
# '2.0.1+rocm5.4.2'
layer = torch.nn.Conv2d(1, 768, kernel_size=16, stride=10)
x = torch.rand(10, 1, 128, 66)
layer.to('cuda:0')(x.to('cuda:0')).sum()
# tensor(10433.3145, device='cuda:0', grad_fn=<SumBackward0>)
layer.to('cpu')(x.to('cpu')).sum()
# tensor(1619.2983, grad_fn=<SumBackward0>)

import torch
torch.__version__
'2.0.0'
layer = torch.nn.Conv2d(1, 768, kernel_size=16, stride=10)
x = torch.rand(10, 1, 128, 66)
layer.to('cuda:0')(x.to('cuda:0')).sum()
# tensor(244.4510, device='cuda:0', grad_fn=<SumBackward0>)
layer.to('cpu')(x.to('cpu')).sum()
# tensor(244.4510, grad_fn=<SumBackward0>)
# `torch.allclose` returns True