import torch
import torch.nn as nn

class Conv3d(nn.Module):

    def __init__(self):
        super(Conv3d, self).__init__()
        self.conv3d = nn.Conv3d(8, 8, (1, 1, 1), stride=(1, 1, 1), padding=(1, 1, 1), groups=8, dilation=(1, 1, 1), bias=True)

    def forward(self, x):
        return self.conv3d(x)

input = torch.randn(1, 8, 1, 10, 10)
model = Conv3d()

torch.onnx.export(model, input, "./dd.onnx", export_params=False)