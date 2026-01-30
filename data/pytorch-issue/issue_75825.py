import torch.nn as nn

import torch
from torch.ao.quantization.quantize_fx import prepare_fx

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv1d(3, 3, 3)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

m = Model().eval()
qconfig_dict = torch.ao.quantization.get_default_qconfig_dict("fbgemm")
prepare_fx(m, qconfig_dict)

# Workaround:
# qconfig_dict = {"": torch.ao.quantization.get_default_qconfig("fbgemm")}
# prepare_fx(m, qconfig_dict)