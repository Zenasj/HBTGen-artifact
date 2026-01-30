import torch.nn as nn

py
import torch
from torch.func import jacrev
from torch.autograd.functional import jacobian


torch.manual_seed(420)

model_input = torch.ones((1,1,1,1,0))

def func(model_input):
    layer = torch.nn.PixelShuffle(1)
    pred = layer(model_input)
    return pred


print(func(model_input))
# tensor([], size=(1, 1, 1, 1, 0))

jacrev(func)(model_input)
# CRASH: floating point exception (core dumped)

jacobian(func, model_input)
# RuntimeError: stack expects a non-empty TensorList