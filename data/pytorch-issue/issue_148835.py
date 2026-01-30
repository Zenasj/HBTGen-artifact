import torch.nn as nn

import torch
torch.set_grad_enabled(True)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.relu_(x)
        return x

model = SimpleModel()
inputs = torch.randn(1, 10)
compiled_model = torch.compile(model, backend='inductor')
compiled_out = compiled_model(*inputs)