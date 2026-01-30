import torch

grad_output = torch.full((5, 4, 5, 1, 3,), 0.320139, dtype=torch.double)
self = torch.full((5, 4, 4, 1, 3,), 7.29222, dtype=torch.double)
indices = torch.full((5, 4, 1, 3,), 5, dtype=torch.long)

grad_input = torch.ops.aten.adaptive_max_pool3d_backward(grad_output, self, indices)