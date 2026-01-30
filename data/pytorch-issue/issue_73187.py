import torch

grad_output = torch.full((1, 1, 1, 4, 4,), 1, dtype=torch.float64, requires_grad=True)
input = torch.full((5, 5, 5, 5, 5,), 3.5e+35, dtype=torch.float64, requires_grad=True)
grid = torch.full((1, 1, 1, 4, 4,), 1, dtype=torch.float64, requires_grad=True)
interpolation_mode = 0
padding_mode = 0
align_corners = True
res = torch.grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners)
grad_out = torch.zeros_like(res)
torch.autograd.backward(res, grad_tensors=grad_out)