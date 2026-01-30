import torch.nn as nn

import torch
from torch import nn as nn

class CustomModule(nn.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        # Define a buffer
        self.register_buffer('my_buffer', torch.tensor(4.0))

    def forward(self, x1, x2):
        # Mutate the buffer
        self.my_buffer.add_(1.0) # In-place addition

        return torch.nn.functional.relu(x1 + x2)

inputs = (torch.randn(3), torch.randn(3))

export_output = torch.onnx.dynamo_export(CustomModule(), *inputs)