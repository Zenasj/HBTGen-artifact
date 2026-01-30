import torch.nn as nn

import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_feature = 16
        self.oh = 4
        self.ow = 8
        self.out_feature = self.oh * self.ow
        self.linear = torch.nn.Linear(self.in_feature, self.out_feature)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.linear(x)
        y = self.relu(y)
        return y.view(y.size(0), 1, self.oh, self.ow)
        # return y.view(x.size(0), 1, self.oh, self.ow) # This gives a similar result

model_fp32 = M().eval()

qengine = 'fbgemm'
torch.backends.quantized.engine = qengine
qconfig_mapping = get_default_qconfig_mapping(qengine)
x = torch.randn((5, 16))
prepared_model = prepare_fx(model_fp32, qconfig_mapping, x)
prepared_model(x)
quantized_model = convert_fx(prepared_model)
print(quantized_model)