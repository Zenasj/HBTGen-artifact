import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 24, 3)

    def forward(self, x):
        x = self.conv(x)
        return x

class Net_slice(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 24, 3)

    def forward(self, x):
        x = self.conv(x)
        x1 = x[:, 0:12, :, :]
        x2 = x[:, 12:24, :, :]
        return x1, x2


x = torch.randn(1, 3, 32, 32)
model = Net()
model_slice = Net_slice()

print('###########################')
print('model without slice')
torch.onnx.export(model, x, "export.onnx", verbose=True)

print('###########################')
print('model with slice')
torch.onnx.export(model_slice, x, "export.onnx", verbose=True)