import torch

grad_output = torch.full((1, 4, 4, 4,), 0.320139, dtype=torch.float)
self = torch.full((1, 4, 4, 4,), 7.29222, dtype=torch.float)
indices = torch.full((1, 4, 3, 1,), 5, dtype=torch.long)

grad_input = torch.ops.aten.adaptive_max_pool2d_backward(grad_output, self, indices)