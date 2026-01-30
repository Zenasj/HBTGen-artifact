import torch.nn as nn

import torch
import torch.quantization.quantize_fx as quantize_fx
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        return self.tanh(x)


model_fp32 = M()
model_fp32.eval()
x = torch.randn((1,2,3,4));

qconfig = get_default_qconfig('fbgemm')
qconfig_dict = {"": qconfig}
prepared_model = prepare_fx(model_fp32, qconfig_dict, x) # error occurs here
quantized_model = convert_fx(prepared_model)
print(quantized_model)
out = quantized_model(x)