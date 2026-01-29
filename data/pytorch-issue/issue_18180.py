# torch.rand(1, 3, dtype=torch.float)  # Inferred input shape for the model

import torch
import torch.nn as nn


class Child(nn.Module):
    def __init__(self, lin_in, linear, lin_out):
        super().__init__()
        self.lin_in = lin_in
        self.linear = linear
        self.lin_out = lin_out

    def forward(self, x):
        return self.lin_out(self.linear(self.lin_in(x)))


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linears = nn.ModuleList([
            nn.Linear(2000, 5000)  # 38.14MiB
            for _ in range(10)
        ])
        self.lin_in = nn.Linear(3, 2000)
        self.lin_out = nn.Linear(5000, 4)

    @property
    def num_linears(self):
        return len(self.linears)

    def get_child(self, i):
        return Child(self.lin_in, self.linears[i % self.num_linears], self.lin_out)

    def forward(self, x, child_index=0):
        child = self.get_child(child_index)
        return child(x)


def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, dtype=torch.float)

