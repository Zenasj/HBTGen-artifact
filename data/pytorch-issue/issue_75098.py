import torch.nn as nn

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        weight = torch.rand(32, 3, 3, 3, dtype=torch.float32)
        self.weight = nn.Parameter(weight)
    
    def forward(self, x):
        return torch._convolution(x, self.weight, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1, False, False, True)

model = Model()
model.eval()

batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
model_out = model(x)

torch.onnx.export(
    model, x, 'conv.onnx',
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'},
    },
    verbose=True
)