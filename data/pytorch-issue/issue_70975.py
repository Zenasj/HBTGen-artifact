import torch
a = torch.rand([2, 2])
a.ravel()
# succeed
b = torch.rand([2, 2])
a.ravel(b)
# TypeError: _TensorBase.ravel() takes no arguments (1 given)