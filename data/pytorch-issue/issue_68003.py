import torch.nn as nn

import torch
from torch import nn


class IndexSelectModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('channel_order', torch.tensor([2, 1, 0]))

    def forward(self, x):
        # x1 = x[:, [2, 1, 0], :, :]
        x1 = torch.index_select(x, 1, self.channel_order)
        return x1


class MaxPoolModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('channel_order', torch.tensor([2, 1, 0]))
        self.avgpool = nn.AdaptiveMaxPool2d((4, 4))

    def forward(self, x):
        # x1 = x[:, [2, 1, 0], :, :]
        x1 = torch.index_select(x, 1, self.channel_order)
        x2 = self.avgpool(x1)
        return x2


def to_onnx(m, name):
    dummy = torch.rand((2, 3, 16, 16))
    with torch.no_grad():
        torch.onnx.export(m, dummy,
                          f'{name}.pytorch-{torch.__version__}.onnx',
                          input_names=['input'],
                          output_names=['output'],
                          do_constant_folding=True,
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})



def main():
    m1 = IndexSelectModule()
    m2 = MaxPoolModule()
    m1.eval()
    m2.eval()

    to_onnx(m1, 'index_select')
    to_onnx(m2, 'max_pool')


if __name__ == '__main__':
    main()