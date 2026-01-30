import torch
import torch.nn as nn

class Example(nn.Module):
    def __init__(self, return_duplicate):
        super(Example, self).__init__()
        self.return_duplicate = return_duplicate

    def forward(self, x):
        return x if not self.return_duplicate else 2*x,x

n = Example(False)
x = torch.randn(10)
y = n(x)
print(x)
print(y)

tensor([-0.62339205, -0.99053943,  1.29613805, -0.02801818, -1.50209475,
         0.26496911, -1.35937905,  0.03782356,  0.15036489,  1.21293926])
(tensor([-0.62339205, -0.99053943,  1.29613805, -0.02801818, -1.50209475,
         0.26496911, -1.35937905,  0.03782356,  0.15036489,  1.21293926]), tensor([-0.62339205, -0.99053943,  1.29613805, -0.02801818, -1.50209475,
         0.26496911, -1.35937905,  0.03782356,  0.15036489,  1.21293926]))

self.return_duplicate

x,x