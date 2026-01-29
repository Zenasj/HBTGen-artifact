# torch.rand(B, D, dtype=torch.float32)  # B=batch_size (3), D=input dimension (5)
import torch
import torch.nn as nn
import torch.nn.functional as F

class FunctionalDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=self.training)

class ModuleDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)

class MyModel(nn.Module):
    def __init__(self, dropout_p=1.0):
        super().__init__()
        self.functional = FunctionalDropout(dropout_p)
        self.module = ModuleDropout(dropout_p)

    def forward(self, x):
        # Returns tuple of outputs from both implementations
        # Functional path may throw ZeroDivisionError when compiled with p=1
        return (self.functional(x), self.module(x))

def my_model_function():
    # Creates model with p=1 (problematic case)
    return MyModel()

def GetInput():
    # Matches input shape from issue's second example (batch_size=3, input_dim=5)
    return torch.rand(3, 5, dtype=torch.float32)

