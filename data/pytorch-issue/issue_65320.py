import torch.nn as nn

import torch
from torch import nn
import torch.nn.functional as F
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)

    def forward(self, x):
        x = self.conv(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)))
        return x

float_model = M()

float_model.eval()
qconfig = get_default_qconfig("fbgemm")
qconfig_dict = {"": qconfig}

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)

prepared_model = prepare_fx(float_model, qconfig_dict)  # fuse modules and insert observers
#calibrate(prepared_model, data_loader_test)  # run calibration on sample data
quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model