import torch
import torch.nn as nn

affine_matrix = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]], dtype=torch.float)
grid = torch.nn.functional.affine_grid(affine_matrix, (1,1,3,3,3))