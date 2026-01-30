# test_instance_norm_export.py

import torch
import torch.nn as nn
from torch.onnx import export


class Model(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.layer = nn.InstanceNorm2d(num_features=channels)

    def forward(self, x):
        return self.layer(x)


def main():
    dummy_input = torch.randn(1, 3, 960, 960)

    model = Model()
    model.eval()

    with torch.no_grad():
        export(model, dummy_input, 'test.onnx', opset_version=11)


if __name__ == '__main__':
    main()