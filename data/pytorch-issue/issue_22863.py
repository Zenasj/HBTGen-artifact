import torch
print("torch version:", torch.__version__)
import torch.nn as nn

class Somenet(nn.Module):
    def __init__(self):
        super(Somenet, self).__init__()

        self.conv1 = nn.Conv2d(90, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2[:, :x1.size(1)] = x2[:, :x1.size(1)] + x1
        x3 = self.conv3(x2)
        return x3

x = torch.randn(3, 90, 5, 6)
torch.onnx.export(Somenet(), (x,), "/dev/null", verbose=True, opset_version=11)