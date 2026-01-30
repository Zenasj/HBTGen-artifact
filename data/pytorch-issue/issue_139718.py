import torchvision

import torch

from torchvision.models.resnet import resnet18
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)
from torch.ao.quantization.quantize_pt2e import (
  prepare_pt2e,
  convert_pt2e,
)
from executorch import exir

model = resnet18(pretrained=True)
model.to("cpu").eval()
example_inputs = (torch.rand(1, 3, 224, 224),)
exported_model = torch.export.export_for_training(model, example_inputs).module()

quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config(
    is_per_channel = True,
    is_dynamic = False,
    act_qmin = -128,
    act_qmax = 127,
    weight_qmin = -127,
    weight_qmax = 127)
)
prepared_model = prepare_pt2e(exported_model, quantizer)
quantized_model = convert_pt2e(prepared_model, use_reference_representation=True)

aten_dialect_program = torch.export.export(quantized_model, example_inputs)
edge_dialect_program = exir.to_edge(aten_dialect_program)
executorch_program = edge_dialect_program.to_executorch()
with open(f"exported/r18_quantized.pte", "wb") as f:
    f.write(executorch_program.buffer)