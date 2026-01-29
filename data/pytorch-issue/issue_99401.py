# torch.rand(1).bool(), torch.rand(1, 3)  # pred and x inputs
import torch
import torch.nn as nn

def cond(pred, true_fn, false_fn, args):
    if pred.item():
        return true_fn(*args)
    else:
        return false_fn(*args)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, inputs):
        pred, x = inputs
        y = x + x

        def true_fn(val):
            return self.linear(val) * (x + y)

        def false_fn(val):
            return val * (y - x)

        return cond(pred, true_fn, false_fn, (x,))

def my_model_function():
    return MyModel()

def GetInput():
    pred = torch.rand(1).bool()
    x = torch.rand(1, 3)
    return (pred, x)

