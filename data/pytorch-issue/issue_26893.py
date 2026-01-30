import torch.nn as nn

import torch
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()

    def forward(self, inputs):
        return torch.einsum('i->i',inputs)

model = MyModel()
torch.onnx.export(
    model,
    torch.randn(1),
    "pyhf.onnx",
    verbose=True,
    input_names=['input'],
    output_names=['output']
)