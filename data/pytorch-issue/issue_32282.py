import torch.nn as nn

python
import torch
from torch import nn

class ExportModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # n, c, h, w = x.shape
        # y = nn.functional.layer_norm(x, [c, h, w])       # not working
        # y = nn.functional.layer_norm(x, x.size()[1:])     # not working
        y = nn.functional.layer_norm(x, [16, 32, 128])

        return y

def main():
    model = ExportModel()

    dummy_input = torch.randn(64, 16, 32, 128)
    input_names = [ "input" ]
    output_names = [ "output" ]

    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, "sample.onnx", verbose=True,
            input_names=input_names, output_names=output_names
        )
    return

if __name__ == '__main__':
    main()