import torch
import torch.nn as nn


class Unfold1(nn.Module):
    def forward(self, x):
        return x.unfold(-1, 4, 2)

model = Unfold1()

x = torch.arange(72).reshape(1, 3, 24).float()
y = model(x)
print(y.shape)

torch.onnx.export(
    model, x, 'model.onnx', verbose=False,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch', 1: 'time', 2: 'freq'}}, opset_version=11
)

dynamic_axes={'input': {0: 'batch', 1: 'time'}}