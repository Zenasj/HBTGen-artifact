import torch
from torch import tensor as tensor
t = tensor([[11.1250],
         [25.5000],
         [21.3125],
         [11.6875],
         [35.8750],
         [44.8125],
         [46.3125],
         [36.1875],
         [38.8125],
         [43.6875],
         [43.8750],
         [36.1875]])
print(torch.tanh(t))

import torch
from torch import tensor as tensor

torch.set_default_tensor_type(torch.DoubleTensor)
t = tensor([[11.1250],
         [25.5000],
         [21.3125],
         [11.6875],
         [35.8750],
         [44.8125],
         [46.3125],
         [36.1875],
         [38.8125],
         [43.6875],
         [43.8750],
         [36.1875]])
print(torch.tanh(t))

tensor([[1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]])