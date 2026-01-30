import torch
import torch.nn as nn

print(torch.__version__)

class M(torch.nn.Module):
    def forward(self, x):
        x = torch.sort(x)
        return x

m = M().eval()
mp = torch.quantization.quantize_fx.prepare_fx(m, {'': torch.quantization.default_qconfig})
mq = torch.quantization.quantize_fx.convert_fx(mp)
mqs = torch.jit.script(mq) # this line fails