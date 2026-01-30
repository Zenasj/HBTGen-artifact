import torch

subnet.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(subnet, inplace=True)
print(torch.backends.quantized.supported_engines)
torch.quantization.convert(subnet, inplace=True)
script_subnet = torch.jit.script(model)
script_subnet_optimized = optimize_for_model(script_model)