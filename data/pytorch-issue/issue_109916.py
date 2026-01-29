# torch.rand(3, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class LambdaSub(nn.Module):
    def __init__(self, is_true):
        super().__init__()
        self.is_true = is_true

    def forward(self, x):
        return x - x.cos() if self.is_true else x + x.sin()

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.true_graph = LambdaSub(True)
        self.false_graph = LambdaSub(False)

    def forward(self, x):
        cond = (x.shape[0] == 4)  # Symbolic condition for Dynamo's higher_order.cond
        return torch.ops.higher_order.cond(
            cond, self.true_graph, self.false_graph, [x]
        )[0]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 4, dtype=torch.float32)

