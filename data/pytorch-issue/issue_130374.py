import torch.nn as nn

import torch

class Linear(torch.nn.Module):
    def __init__(self, input_shape: int = 784):
        super().__init__()
        self.input_shape = input_shape
        self.fc = torch.nn.Linear(input_shape, 10)

    def forward(self, x):
        x = x.view(
            x.shape[0], self.input_shape
        )  # num samples is first dim. Then flatten the rest
        x = self.fc(x)
        return x

sample_input = {'x': torch.randn(10, 784, 1) }
input_names = ['x']
dynamic_axes = { 'x': [0] }
model = Linear()

torch.onnx.export(
    model,
    args=(sample_input,),
    f="/tmp/traced.onnx",
    training=torch.onnx.TrainingMode.TRAINING,
    do_constant_folding=False,
    input_names=input_names,
    dynamic_axes=dynamic_axes
)