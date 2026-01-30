import torch.nn as nn

import torch
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.fx.experimental import normalize

class TestModule(torch.nn.Module):
    def forward(self, x):
        return x + x

mod = TestModule()
mod.eval()
config = {"": torch.quantization.get_default_qconfig("fbgemm")}
mod = prepare_fx(mod, config)
mod = convert_fx(mod)
mod = torch.fx.Transformer(mod).transform()