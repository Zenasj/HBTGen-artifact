import torchvision

import torch
import io
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torch.ao.quantization import quantize_fx

# code similar to the example given in https://pytorch.org/docs/stable/quantization.html#saving-and-loading-quantized-models

model1 = mobilenet_v3_large(weights='DEFAULT').eval()

model_to_quantize=model1.eval()
model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, torch.ao.quantization.get_default_qat_qconfig_mapping('fbgemm'), torch.randn(1,3,224,224))
[model_prepared(torch.randn(1,3,224,224)) for _ in range(5)] # calibrate
model_quantized_orig = quantize_fx.convert_fx(model_prepared)

b = io.BytesIO()
torch.save(model_quantized_orig.state_dict(), b)

model2 = mobilenet_v3_large(weights='DEFAULT').eval()
model_to_quantize=model2.eval()
model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, torch.ao.quantization.get_default_qat_qconfig_mapping('fbgemm'),torch.randn(1,3,224,224))
model_prepared.to("cpu")
model_quantized_new = quantize_fx.convert_fx(model_prepared)
b.seek(0)
loaded = torch.load(b)
m = model_quantized_new.load_state_dict(loaded)
print(m)
print(model_quantized_orig.get_submodule('features.0.2').scale,model_quantized_new.get_submodule('features.0.2').scale) # 'features.0.2' is a QuantizedHardSwish layer

self.scale = scale
self.zero_point=zero_point

self.register_buffer('scale',scale)
self.register_buffer('zero_point',zero_point)