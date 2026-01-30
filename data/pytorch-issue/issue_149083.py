import torch.nn as nn

py
import math
import torch

class Gray(torch.nn.Module):
    nbits: int = 32

    def forward(self, gray: torch.Tensor):
        shifts = [(0x1 << i) for i in range((math.ceil(math.log(self.nbits, 2)) - 1), -1, -1)]
        for shift in shifts:
            gray ^= gray >> shift

        return gray


torch.onnx.export(
    Gray(),  # model to export
    (torch.randint(0, 100, [100], dtype=torch.long)),  # inputs of the model,
    "my_model.onnx",  # filename of the ONNX model
    dynamo=True,  # True or False to select the exporter to use
    verbose=False,
)