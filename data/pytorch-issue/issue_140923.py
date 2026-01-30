import torch

grad_output = torch.full((9,7,9,9,), 4.91269, dtype=torch.float)
self = torch.full((5,7,9,9,), 9.71351, dtype=torch.float)
kernel_size = [3,3]
stride = [1,1]
padding = [1,1]
dilation = [1,1]
ceil_mode = False
indices = torch.full((5,7,9,9,), 5, dtype=torch.long)
torch.ops.aten.max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices)