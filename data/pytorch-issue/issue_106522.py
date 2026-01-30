import torch.nn as nn

print(torch.backends.quantized.supported_engines)  # outputs: ['none', 'fbgemm']

#!/usr/bin/env python
import sys
import os
import torch
import torch.utils.bundled_inputs
import torch.utils.mobile_optimizer
import torch.backends._nnapi.prepare
import torchvision.models.quantization.mobilenet
from pathlib import Path

def make_mobilenetv2_nnapi(output_dir_path, quantize_mode):
    quantize_core, quantize_iface = {
        "none": (False, False),
        "core": (True, False),
        "full": (True, True),
    }[quantize_mode]

    model = torchvision.models.quantization.mobilenet.mobilenet_v2(pretrained=True, quantize=quantize_core)
    model.eval()

    if not quantize_core:
        model.fuse_model()
    assert type(model.classifier[0]) == torch.nn.Dropout
    model.classifier[0] = torch.nn.Identity()

    input_float = torch.zeros(1, 3, 224, 224)
    input_tensor = input_float

    if quantize_core:
        quantizer = model.quant
        dequantizer = model.dequant
        model.quant = torch.nn.Identity()
        model.dequant = torch.nn.Identity()
        input_tensor = quantizer(input_float)

    input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
    input_tensor.nnapi_nhwc = True

    with torch.no_grad():
        traced = torch.jit.trace(model, input_tensor)
    nnapi_model = torch.backends._nnapi.prepare.convert_model_to_nnapi(traced, input_tensor)

    if quantize_core and not quantize_iface:
        nnapi_model = torch.nn.Sequential(quantizer, nnapi_model, dequantizer)
        model.quant = quantizer
        model.dequant = dequantizer
        # Switch back to float input for benchmarking.
        input_tensor = input_float.contiguous(memory_format=torch.channels_last)

    model = torch.utils.mobile_optimizer.optimize_for_mobile(torch.jit.script(model))

    class BundleWrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, arg):
            return self.mod(arg)

    nnapi_model = torch.jit.script(BundleWrapper(nnapi_model))
    torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
        model, [(torch.utils.bundled_inputs.bundle_large_tensor(input_tensor),)])
    torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
        nnapi_model, [(torch.utils.bundled_inputs.bundle_large_tensor(input_tensor),)])

    # Save both models.
    model._save_for_lite_interpreter(str(output_dir_path / ("mobilenetv2-quant_{}-cpu.pt".format(quantize_mode))))
    nnapi_model._save_for_lite_interpreter(
        str(output_dir_path / ("mobilenetv2-quant_{}-nnapi.pt".format(quantize_mode))))


if __name__ == "__main__":
    for quantize_mode in ["none", "core", "full"]:
        make_mobilenetv2_nnapi(Path(os.path.curdir) / "mobilenetv2-nnapi", quantize_mode)