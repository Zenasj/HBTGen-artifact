import torch

grad_output = torch.full((9, 1, 1, 0, 3,), 0, dtype=torch.double)
self = torch.full((10, 10, 7, 8, 1, 7, 4, 10, 10,), 1.23457e+13, dtype=torch.double)
kernel_size = [-1, -1]
output_size = [1, 1]
indices = torch.full((9, 6, 0, 9, 2, 1, 0, 2,), 0, dtype=torch.long)

torch.ops.aten.fractional_max_pool2d_backward(grad_output, self, kernel_size, output_size, indices)