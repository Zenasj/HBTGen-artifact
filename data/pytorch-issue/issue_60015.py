import torch.nn as nn

#! /usr/bin/python3

import torch
import torch.autograd.profiler as profiler

class InterpTest(torch.nn.Module):
    def __init__(self):
        super(InterpTest, self).__init__()
        self.interp = torch.nn.functional.interpolate
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.interp(x, size=(600, 800), mode='bilinear')
        return self.dequant(x)

model = InterpTest()
model.eval()
torch.backends.quantized.engine = 'qnnpack'
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model, inplace=True)
test_input = torch.rand(1, 3, 720, 1280)
model(test_input)

with torch.no_grad():
    with profiler.profile(record_shapes=False) as prof:
        with profiler.record_function("model_inference"):
            model(trace_input)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))