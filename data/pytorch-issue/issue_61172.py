import torch.nn as nn

class AdaptiveConcatPool1d(nn.Module):
    """Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`.
    Adapted from:
    https://fastai1.fast.ai/layers.html#AdaptiveConcatPool2d
    https://github.com/fastai/fastai1/blob/master/fastai/layers.py#L176
    """

    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool1d(self.output_size)
        self.mp = nn.AdaptiveMaxPool1d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 2)

input_sample = torch.randn((1, 2, 130))
model.to_onnx(
        file_path='model.onnx',
        input_sample=input_sample,
        export_params=True,
    )

import torch
from torch import nn

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    _layer_size = 23
    self.net = nn.Sequential(
        nn.Linear(10, 100),
        nn.AdaptiveMaxPool1d(_layer_size)
        )

  def forward(self, x):
    return self.net(x)

net = Net()
net.eval()

x = torch.randn((1, 1, 10))
y = net(x)
print(f"x shape: {x.shape}, y.shape: {y.shape}")

torch.onnx.export(net, x, "main.onnx")