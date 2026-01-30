import torch
from torch.onnx import export as onnx_export
import torch.nn as nn
import torch.nn.functional as F

class InstanceNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input):
        output = F.instance_norm(input)
        return output

instancenorm = InstanceNorm().eval()

random_input = torch.randn(1, 2, 2)

output = instancenorm(random_input)
print(output)


input_names = [ "input_1" ]
output_names = [ "output1" ]

with torch.no_grad():
  onnx_export(
              instancenorm,
              random_input,
              f="instancenorm_test.onnx",
              input_names=input_names,
              output_names=output_names,
              dynamic_axes=None,
              do_constant_folding=False,
              opset_version=11,
              verbose=True
          )

import torch
from torch.onnx import export as onnx_export
import torch.nn as nn
import torch.nn.functional as F

class InstanceNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input):
        output = F.instance_norm(input, use_input_stats=False, running_mean=torch.zeros(2), running_var=torch.ones(2))
        return output

instancenorm = InstanceNorm().eval()

random_input = torch.randn(1, 2, 2)

output = instancenorm(random_input)
print(output)


input_names = [ "input_1" ]
output_names = [ "output1" ]

with torch.no_grad():
  onnx_export(
              instancenorm,
              random_input,
              f="instancenorm_test.onnx",
              input_names=input_names,
              output_names=output_names,
              dynamic_axes=None,
              do_constant_folding=False,
              opset_version=11,
              verbose=True
          )