import torch.nn as nn

import torch
import torchvision.transforms as T
from torch import nn


class CropNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "CropNet"

    def forward(self, x):
        _, _, sx, sy = x.size()
        x = T.CenterCrop((sx // 2, sy // 2))(x)
        return x

    def save_onnx(self, path, verbose):
        dummy_input = torch.randn(8, 1, 64, 64)
        torch.onnx.export(self, dummy_input, path, verbose=verbose,
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={
                              "input": {0: "batch"},
                              "output": {0: "batch"}
                          },
                          )


if __name__ == "__main__":
    net = CropNet()
    cropped = net.forward(torch.randn(8, 1, 64, 64))  # normal execution works
    net.save_onnx("cropnet.onnx", False)  # export gives error message

import torch
import torchvision.transforms as T
from torch import nn
import onnx


class CropNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "CropNet"

    def forward(self, x):
        _, _, sx, sy = x.size()
        x = T.CenterCrop((sx // 2, sy // 2))(x)
        return x


net = CropNet()
dummy_input = torch.randn(8, 1, 64, 64)
exported = torch.onnx.dynamo_export(net, dummy_input)
print(onnx.printer.to_text(exported.model_proto))