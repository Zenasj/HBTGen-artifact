import torch.nn as nn

import torch
import torch._dynamo
import logging

from torch._dynamo import config

config.verbose = True
# config.log_level = logging.INFO

class Model(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.norm = torch.nn.InstanceNorm3d(num_features=1)
        self.conv = torch.nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3)
        self.activation = torch.nn.ReLU(inplace=True)
        self.final = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        y = self.norm(x)
        y = self.conv(y)
        y = self.activation(y)
        y = self.final(y)
        return y


def main():
    model = Model()
    model = model.float()
    device = torch.device('cuda:0')
    model = model.to(device)
    opt_model = torch.compile(model, backend='inductor')

    input_shape = (1, 64, 64, 64)
    x = torch.randn(input_shape, dtype=torch.float32, device=device)

    result = opt_model(x)
    print(result.shape)


if __name__ == '__main__':
    main()

torch.Size([3, 60, 60, 60])