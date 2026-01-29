# Input is a tuple (pred1 (scalar bool), x1 (3,3), pred2 (scalar bool), x2 (3), x3 (3))
import torch
from functorch.experimental.control_flow import cond
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)  # From first example's model
        self.f1 = lambda x1, x2: x1 + x2  # From second example's f1
        self.f2 = lambda x1, x2: x1 * x2  # From second example's f2

    def forward(self, inputs):
        pred1, x1, pred2, x2, x3 = inputs
        # First condition (linear model)
        def true_fn1(val):
            return self.linear(val) * 2
        def false_fn1(val):
            return self.linear(val) * -1
        res1 = cond(pred1, true_fn1, false_fn1, [x1])

        # Second condition (f1/f2)
        res2 = cond(pred2, self.f1, self.f2, [x2, x3])

        return (res1, res2)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate inputs for both conditions
    x1 = torch.randn(3, 3)
    pred1 = torch.tensor(x1[0, 0].item() < 0)
    x2 = torch.randn(3)
    x3 = torch.randn(3)
    pred2 = torch.tensor(True)  # Fixed for reproducibility
    return (pred1, x1, pred2, x2, x3)

