import torch.nn as nn

import torch

class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.max(x, x + 1)

net = Net()
onnx_model = 'test.onnx'

torch.onnx.export(net, (torch.zeros((3, 3), dtype=torch.int32),),
                  onnx_model, verbose=True, opset_version=11)