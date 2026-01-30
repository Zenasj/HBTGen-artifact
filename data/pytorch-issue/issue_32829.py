import torch
import torch.nn as nn

# example code to reproduce the bug:
image = torch.arange(0, 5, dtype=torch.float).expand((1,1,5,5)).requires_grad_()
id_grid = torch.nn.functional.affine_grid(
    torch.tensor([[[1,0,0],[0,1,0.]]]), (1,1,5,5), align_corners=True).requires_grad_()
torch.nn.functional.grid_sample(image, id_grid, padding_mode='border',
                                align_corners=True).sum().backward()
print(id_grid.grad.permute(0,3,1,2))

tensor([[[[ 2.,  2.,  2.,  2., -8.],
          [ 2.,  2.,  2.,  2., -8.],
          [ 2.,  2.,  2.,  2., -8.],
          [ 2.,  2.,  2.,  2., -8.],
          [ 2.,  2.,  2.,  2., -8.]],

         [[ 0.,  0.,  0.,  0.,  0.],
          [ 0.,  0.,  0.,  0.,  0.],
          [ 0.,  0.,  0.,  0.,  0.],
          [ 0.,  0.,  0.,  0.,  0.],
          [ 0., -2., -4., -6., -8.]]]])

tensor([[[[0., 2., 2., 2., -0.],
          [0., 2., 2., 2., -0.],
          [0., 2., 2., 2., -0.],
          [0., 2., 2., 2., -0.],
          [0., 2., 2., 2., -0.]],

         [[0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., -0., -0., -0., -0.]]]])