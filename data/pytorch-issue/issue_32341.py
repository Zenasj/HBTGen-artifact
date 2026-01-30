import torch.nn as nn

nn.GroupNorm(32, 128)

import torch
from torch import nn
try:
    from alfred.dl.torch.common import device
except ImportError:
    print('install alfred-py first, pip install alfred-py')

torch.manual_seed(1024)
model_p = 'model_instancenorm.onnx'


class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.expander = nn.Conv2d(3, 192, 1, 1)
        self.channel_c = nn.Conv2d(192, 128, 1)
        self.l = nn.GroupNorm(32, 128)

    def forward(self, x: torch.Tensor):
        x = self.expander(x)
        x = self.channel_c(x)
        print(x.shape)
        x = self.l(x)
        print(x.shape)
        return x


def export_onnx():
    model = TinyModel().to(device)
    sample_input = torch.rand(1, 3, 544, 1920).to(device)
    model.eval()
    torch.onnx.export(model, sample_input, model_p, input_names=[
                      'img'], output_names=['output'])
    print('onnx model exported. forward now...')
    # forward now


if __name__ == "__main__":
    export_onnx()