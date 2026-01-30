import torch.nn as nn

import torch

torch.set_printoptions(precision=10)
for i in range(10):
    x = torch.randn((10, 3, 100, 100))
    nelem = 10 * 100 * 100
    mean = x.sum(dim=(0, 2, 3)) / nelem
    var = torch.sum((x - mean[None, :, None, None]) ** 2, dim=(0, 2, 3)) / nelem

    torch_var = x.var(dim=(0, 2, 3), unbiased=False)
    torch_mean = x.mean(dim=(0, 2, 3))
    print(i)
    print(f"means are exactly equal {torch.all(mean == torch_mean)}")
    print(f"vars are exactly equal {torch.all(var == torch_var)}")

import torch
from torch import nn

class CustomBatchNorm(nn.BatchNorm2d):

    def forward(self, input):
        self._check_input_dim(input)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        B, C, H, W = input.shape
        if self.training:
            n = input.shape[0] * input.shape[2] * input.shape[3]
            mean = torch.sum(input, dim=(0, 2, 3)) / n
            var = (input - mean.view(1, C, 1, 1)) ** 2
            var = var.sum(dim=(0, 2, 3)) / n

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) + \
                                   (1 - exponential_average_factor) * self.running_var

        else:
            mean = self.running_mean
            var = self.running_var

        # Norm the input
        normed = (input - mean.view(1, C, 1, 1)) / (torch.sqrt(var + self.eps)).view(1, C, 1, 1)

        # Apply affine parameters
        if self.affine:
            normed = normed * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return normed

my_bn = CustomBatchNorm(3, affine=True)
bn = nn.BatchNorm2d(3, affine=True)

# Run train
for _ in range(10):
    scale = torch.randint(1, 10, (1,)).float()
    bias = torch.randint(-10, 10, (1,)).float()
    x = torch.randn(10, 3, 100, 100) * scale + bias
    out1 = my_bn(x)
    out2 = bn(x)
    print(torch.allclose(out1, out2))

False
False
False
False
False
False
False
False
False
False