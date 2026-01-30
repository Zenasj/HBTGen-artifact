import torch
import torch.nn as nn


class CustomModule(nn.Module):
    def __init__(self, dims):
        super(CustomModule, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x + torch.randn_like(x)

    def random_method(self):
        return torch.randn(self.dims)


if __name__ == '__main__':
    dnn = CustomModule(100)
    out = dnn.random_method()
    print(out)