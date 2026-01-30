import torch
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import torch.nn as nn

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn(5, 5)
        self.b = torch.zeros(5)

    def forward(self, x):
        x = nn.functional.linear(x, self.w, self.b)
        return x

qconfig_dict = {
            "": None,
            "object_typo": [
                (torch.nn.functional.linear, torch.quantization.default_qconfig),
            ]
        }

m = M().eval()

mp = prepare_fx(m, qconfig_dict)
mp(torch.rand(5, 5))
mc = convert_fx(mp)
print(mc)

GraphModule()

def forward(self, x):
    w = self.w
    b = self.b
    linear = torch.nn.functional.linear(x, w, bias = b);  x = w = b = None
    return linear