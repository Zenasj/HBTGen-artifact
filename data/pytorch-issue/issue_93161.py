# test.py
import numpy as np
import torch

mat1 = torch.randn(10, 3, 4)
mat2 = torch.randn(10, 4, 5)
res = torch.bmm(mat1, mat2)
print(res.size())