import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn import functional as F
from torch.quantization import QuantWrapper


class Model(nn.Module):
    def forward(self, x: torch.Tensor):
        return F.interpolate(x, 2, mode="bilinear", align_corners=False)


input = torch.rand(10, 3, 1000, 1000)

model = QuantWrapper(Model())
model.eval()
model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
torch.backends.quantized.engine = "qnnpack"
torch.quantization.prepare(model, inplace=True)
model(input)
torch.quantization.convert(model, inplace=True)
scripted_model = torch.jit.script(model)

with torch.autograd.profiler.profile() as prof:
    scripted_model(input)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))

channels_last_input = input.to(memory_format=torch.channels_last)
with torch.autograd.profiler.profile() as prof:
    scripted_model(channels_last_input)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))