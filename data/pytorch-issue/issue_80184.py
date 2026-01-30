import torch

from torch.ao.quantization.quantize_fx import prepare_fx

model = ModelWithFixedQParamsOps()
qconfig_mapping = QConfigMapping()
example_inputs = ...
prepare_fx(model, qconfig_mapping, example_inputs)

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx

model = ModelWithFixedQParamsOps()
qconfig_mapping = get_default_qconfig_mapping()
example_inputs = ...
prepare_fx(model, qconfig_mapping, example_inputs)