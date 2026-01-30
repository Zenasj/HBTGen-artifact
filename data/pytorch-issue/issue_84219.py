import torch.nn as nn

import torch
import onnxruntime
from onnxruntime.training.ortmodule import DebugOptions, LogLevel
from onnxruntime.training.ortmodule import ORTModule

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = torch.nn.Conv2d(3, 3, 5, 2, 1)

    def forward(self, x):
        x = self.cv1(x)
        return x

x = torch.randn(10, 3, 20, 20) * 2
m = MyModule().eval()
x = x.cuda()
m = m.cuda()

debug_options = DebugOptions(log_level=LogLevel.VERBOSE, save_onnx=True, onnx_prefix="ViT-B")
m = ORTModule(m, debug_options=debug_options)

with torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=True):
    loss = m(x)