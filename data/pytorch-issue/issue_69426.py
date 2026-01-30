import torch
import torchvision
from torch.fx import symbolic_trace
import torch.quantization.quantize_fx as quantize_fx

model = torchvision.models.resnet18().eval()
dummy = torch.randn(1, 3, 224, 224)
model = symbolic_trace(model)

qconfig_dict = {"": torch.quantization.get_default_qconfig('fbgemm')}
prepared_model = quantize_fx.prepare_fx(model, qconfig_dict)
prepared_model(dummy)
quant_module = quantize_fx.convert_fx(prepared_model)

torch.save(quant_module, 'test.pt')
torch.load('test.pt')