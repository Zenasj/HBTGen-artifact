import torch
import torch.nn as nn

class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x

    def get_example_inputs(self):
        return (torch.randn(10),)