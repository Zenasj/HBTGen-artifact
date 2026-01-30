import torch.nn as nn

import torch
from torch import nn


class Model(torch.nn.Module):
    def forward(self, x):
        return nn.functional.interpolate(
            x,
            scale_factor=(x.size(2) / 100, x.size(3) / 200),
            mode="bicubic",
        )


if __name__ == "__main__":
    model = Model()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)

    # This will fail
    torch.jit.trace(model, x)