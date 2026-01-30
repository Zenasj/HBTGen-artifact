import torch.nn as nn

# !pip uninstall --y torch
# !pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
import torch
from torch import nn

device = "cpu"

def to_complex_activation(activation):
    return lambda x: torch.view_as_complex(torch.cat(
        [activation(x.real).unsqueeze(-1), activation(x.imag).unsqueeze(-1)], dim=-1))


class CGCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(CGCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.wg = nn.Parameter(torch.randn(2 * hidden_size, hidden_size, dtype=torch.cfloat))
        self.vg = nn.Parameter(torch.randn(2 * hidden_size, input_size, dtype=torch.cfloat))
        self.bg = nn.Parameter(torch.randn(2 * hidden_size, dtype=torch.cfloat))

        self.w = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=torch.cfloat))
        self.v = nn.Parameter(torch.randn(hidden_size, input_size, dtype=torch.cfloat))
        self.b = nn.Parameter(torch.randn(hidden_size, dtype=torch.cfloat))

        alpha = beta = 0.5  # TODO
        self.fg = lambda x: torch.sigmoid(alpha * x.real + beta * x.imag)
        self.fa = to_complex_activation(torch.sigmoid)

    def _init_hidden(self, x):
        h = torch.zeros((x.shape[0], self.hidden_size),  dtype=torch.cfloat).to(device)

        return h

    def forward(self, x, ht_=None):
        if ht_ is None:
            ht_ = self._init_hidden(x)

        gates = ht_ @ self.wg.T + x @ self.vg.T + self.bg
        g_r, g_z = gates.chunk(2, 1)

        g_r = self.fg(g_r)
        g_z = self.fg(g_z)

        z = (g_r * ht_) @ self.w.T + x @ self.v.T + self.b
        ht = g_z * self.fa(z) + (1 - g_z) * ht_

        return ht

mod = CGCell(2, 2)
inp = torch.rand(3, 2, dtype=torch.cfloat)

out = mod(inp)

out.norm().backward()
# Runs fine