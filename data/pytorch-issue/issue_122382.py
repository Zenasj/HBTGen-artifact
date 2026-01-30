import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1, stride=1, padding=1)
        self.other_tensor = torch.randn(2, 1, 1)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.add(v1, self.other_tensor, alpha=0.875)
        return v2


func = Model().to('cpu')

x = torch.randn(1, 3, 2, 2)

with torch.no_grad():
    print(func(x.clone()))

    func1 = torch.compile(func)
    print(func1(x.clone()))

    print(torch.allclose(func.other_tensor, func1.other_tensor)) # True

"""
tensor([[[[ 1.7728,  1.7728,  1.7728,  1.7728],
          [ 1.7728,  2.1860,  1.7604,  1.7728],
          [ 1.7728,  0.9550,  2.3261,  1.7728],
          [ 1.7728,  1.7728,  1.7728,  1.7728]],

         [[ 0.2579,  0.2579,  0.2579,  0.2579],
          [ 0.2579, -0.5048,  0.6618,  0.2579],
          [ 0.2579,  0.9659,  0.0888,  0.2579],
          [ 0.2579,  0.2579,  0.2579,  0.2579]]]])

tensor([[[[ 1.9624,  1.9624,  1.9624,  1.9624],
          [ 1.9624,  2.3757,  1.9500,  1.9624],
          [ 1.9624,  1.1447,  2.5158,  1.9624],
          [ 1.9624,  1.9624,  1.9624,  1.9624]],

         [[ 0.2639,  0.2639,  0.2639,  0.2639],
          [ 0.2639, -0.4989,  0.6677,  0.2639],
          [ 0.2639,  0.9718,  0.0947,  0.2639],
          [ 0.2639,  0.2639,  0.2639,  0.2639]]]])
"""