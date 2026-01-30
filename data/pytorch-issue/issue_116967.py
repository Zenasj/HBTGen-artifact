import torch.nn as nn

import torch

print(torch.__version__)

class Module(torch.nn.Module):

    def forward(self, x):
        return x.squeeze(dim=[2, 3])

torch.onnx.export(Module(), torch.zeros((2, 3, 1, 1)), "out.onnx", verbose=True)