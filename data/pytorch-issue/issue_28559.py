import torch
import torch.nn as nn

class N(nn.Module):
    __constants__ = ['norm']

    def __init__(self, norm=None):
        super(N, self).__init__()
        self.activation = torch.nn.functional.relu  # Commenting out this line makes it work
        self.norm = norm

    def forward(self, src):
        output = src
        if self.norm is not None:
            output = self.norm(output)
        return output

class M(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_norm = nn.ReLU()
        self.encoder = N(encoder_norm)

    def forward(self, x):
        return self.encoder(x)

torch.jit.script(M())