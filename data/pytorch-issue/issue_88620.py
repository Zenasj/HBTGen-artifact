import torch

from torch.ao.quantization.qconfig import default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx

model = ModelWithFixedQParamsOps()
qconfig_mapping = QConfigMapping().set_global(default_qconfig)
example_inputs = ...
prepare_fx(model, qconfig_mapping, example_inputs)