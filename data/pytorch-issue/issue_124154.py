import torch.nn as nn

import torch

def affine_grid_cl():
    torch.manual_seed(123)
    theta = torch.rand([6, 3, 4], requires_grad=True)
    grid = torch.nn.functional.affine_grid(theta, [6, 1, 3, 5, 5], align_corners=False)
    grad_tensor = torch.rand(grid.shape)
    grad_tensor = grad_tensor.contiguous(memory_format=torch.channels_last_3d)
    grid.backward(grad_tensor)
    return grid, theta.grad

if __name__ == "__main__":
    fwd, bwd_grad = affine_grid_cl()
    print("Test Passed!")