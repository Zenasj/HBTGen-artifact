import torch.nn as nn

import torch


class FoldModel(torch.nn.Module):
    def __init__(self, output_size, kernel_size, stride, padding):
        super(FoldModel, self).__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return torch.nn.functional.fold(x, self.output_size, self.kernel_size, stride=self.stride, padding=self.padding)


inputs = torch.randn(1, 4, 4)
model = FoldModel((4, 4), (2, 2), (2, 2), (0, 0))
res = model(*inputs)
compiled_model = torch.compile(model, backend='inductor')   ### Crash here!!
compiled_out = compiled_model(*inputs)