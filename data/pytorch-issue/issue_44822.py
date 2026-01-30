import torch.nn as nn

import torch


class RFFTLayer(torch.nn.Module):
    def forward(self, x):
        return torch.fft.rfft(x, 1)


torch.onnx.export(RFFTLayer(), torch.rand((1, 1024)), "model.onnx", opset_version=17)