import torch.nn as nn

import torch
from torch import nn


class Demo1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 3)
        self.conv2 = nn.Conv2d(3, 20, 3)

    def forward(self, x):
        return [self.conv1(x), self.conv2(x)]


class Demo2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 3)
        self.conv2 = nn.Conv2d(3, 20, 3)

    def forward(self, x):
        return {'out1': self.conv1(x), 'out2': self.conv2(x)}


def test_demo(model):
    inputs = torch.zeros(2, 3, 512, 512)
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()

    for i in range(100):
        model.train()
        torch.onnx.export(model, inputs,
                          'debug.onnx',
                          input_names=['data'],
                          opset_version=11,
                          verbose=True,
                          dynamic_axes={'data': {0: 'batch', 2: 'width', 3: 'height'}})
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

if __name__ == '__main__':
    model1 = Demo1()
    model2 = Demo2()
    print('demo1')
    test_demo(model1)
    print('demo2')
    test_demo(model2)