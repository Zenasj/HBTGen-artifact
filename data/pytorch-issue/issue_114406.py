import torch.nn as nn

import torch

class CustomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("my_buffer", torch.tensor(4.0))
    def forward(self, x, b):
        output = x + b
        (
            self.my_buffer.add_(1.0) + 3.0
        )  # Mutate buffer through in-place addition
        return output

inputs = (torch.rand((3, 3), dtype=torch.float32), torch.randn(3, 3))

torch.onnx.dynamo_export(CustomModule(), *inputs)

import torch

class CustomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("my_buffer", torch.tensor(4.0))
    def forward(self, x, b):
        output = x + b
        (
            self.my_buffer.add_(1.0) + 3.0
        )  # Mutate buffer through in-place addition
        return output

inputs = (torch.rand((3, 3), dtype=torch.float32), torch.randn(3, 3))

program = torch.onnx.export(CustomModule(), inputs, dynamo=True)
print(program)