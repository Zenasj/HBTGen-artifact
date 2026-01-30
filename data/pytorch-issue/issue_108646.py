import torch.nn as nn

scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
scale = torch.max(scale, self.eps)
zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
zero_point = torch.clamp(zero_point, quant_min, quant_max)

scale = (max_val - min_val) / float(quant_max - quant_min)
scale = torch.max(scale, self.eps)
zero_point = quant_min - torch.round(min_val / scale).to(torch.int)
zero_point = torch.clamp(zero_point, quant_min, quant_max)

# Static quantization of a model consists of the following steps:

#     Fuse modules
#     Insert Quant/DeQuant Stubs
#     Prepare the fused module (insert observers before and after layers)
#     Calibrate the prepared module (pass it representative data)
#     Convert the calibrated module (replace with quantized version)

import torch
from torch import nn
import copy
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver, default_observer
import torch
from torch.ao.quantization.qconfig import QConfig, default_per_channel_qconfig

backend = "fbgemm"  # running on a x86 CPU. Use "qnnpack" if running on ARM.

model = nn.Sequential(
     nn.Linear(100,100),
)

## EAGER MODE
m = copy.deepcopy(model)
m.eval()
"""Fuse
- Inplace fusion replaces the first module in the sequence with the fused module, and the rest with identity modules
"""
# torch.quantization.fuse_modules(m, ['0','1'], inplace=True) # fuse first Conv-ReLU pair
# torch.quantization.fuse_modules(m, ['2','3'], inplace=True) # fuse second Conv-ReLU pair

"""Insert stubs"""
m = nn.Sequential(torch.quantization.QuantStub(), 
                  *m, 
                  torch.quantization.DeQuantStub())

"""Prepare"""
my_qconfig = QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
)

m.qconfig = my_qconfig

torch.quantization.prepare(m, inplace=True)

"""Calibrate
- This example uses random data for convenience. Use representative (validation) data instead.
"""
mi = float("inf")
ma = -float("inf")
with torch.inference_mode():
    for _ in range(10):
        x = torch.randn(100, 100) + 1000 # Expect the 1000 here to not affect quantization scale
        mi = min(mi, x.min())
        ma = max(ma, x.max())
        m(x)

        
"""Convert"""
torch.quantization.convert(m, inplace=True, )
print(m)