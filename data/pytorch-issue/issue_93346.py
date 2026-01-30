import timm
import torch
from torch.quantization import quantize_fx

torch_model = timm.create_model('deit_tiny_patch16_224')
backend = "fbgemm"
qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}

model_prepared = quantize_fx.prepare_fx(torch_model, qconfig_dict)

# We can skip the calibration for the export purpose

model_quantized = quantize_fx.convert_fx(model_prepared)

torch.onnx.export(model_quantized,
                  x, 
                  'model.onnx', 
                  export_params=True, 
                  opset_version=13, 
                  do_constant_folding=False, 
                  input_names=['input'], 
                  output_names=['output'],
                  dynamic_axes={"input":{0:"batch_size"},
                                "output":{0:"batch_size"}}
                  )