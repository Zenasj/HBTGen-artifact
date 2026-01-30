import torch
from torch import linalg

my_tensor = torch.tensor([[-2., -1., 0.],
                          [1., 2., 3.]])
linalg.matrix_norm(A=my_tensor) # Error

import torch
from torch import linalg

my_tensor = torch.tensor([[-2., -1., 0.],
                          [1., 2., 3.]])
linalg.matrix_norm(input=my_tensor)
# tensor(4.3589)