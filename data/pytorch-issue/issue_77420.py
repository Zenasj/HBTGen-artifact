import torch.nn as nn

from torch import nn
import torch.onnx


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv = nn.Conv2d(1, 50, 2)

    # @torch.jit.script
    def forward(self, x):
        out = self.conv(x)
        return out


torch_model = NeuralNetwork()

input = torch.randn(1, 1, 224, 224)

# torch_model = torch.jit.script(torch_model, input)
# torch_model = torch.jit.trace(torch_model, input)
torch.onnx.export(torch_model, input, "model.onnx")