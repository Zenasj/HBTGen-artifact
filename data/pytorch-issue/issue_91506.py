import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def forward(self):
        tensor = torch.linspace(0, 1, 1)

        return tensor

compiled_model = torch.compile(Model())
a = compiled_model()