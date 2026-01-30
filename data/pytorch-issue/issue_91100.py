import torch.nn as nn

import functorch
import torch

functorch.hessian(torch.nn.Softmax(dim=-1))(torch.randn(32, 10))
functorch.hessian(torch.nn.LogSoftmax(dim=-1))(torch.randn(32, 10))