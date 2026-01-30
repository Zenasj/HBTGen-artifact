import torch

input = torch.empty([1,1,2147483650], device='cuda')
input.fill_(0.5)
result=torch._softmax_backward_data(input, input, 2, input)

import torch

input = torch.empty([1,1,2147483650], device='cuda')
input.fill_(0.5)
result=torch._softmax_backward_data(input, input, 2, input.dtype)