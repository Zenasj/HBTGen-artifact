from torchvision import models
import torch
quant_model = models.quantization.resnet50(pretrained=True,quantize=True)
param = [*quant_model.state_dict().values()]

weight = param[0]
per_tensor =  torch._make_per_tensor_quantized_tensor(weight.int_repr(),weight.q_per_channel_scales()[0],weight.q_per_channel_zero_points()[0])
per_channel = torch._make_per_channel_quantized_tensor(weight.int_repr(),weight.q_per_channel_scales(),weight.q_per_channel_zero_points(),axis=0)

per_tensor[0][0][0][0] = 0.0061
print(per_tensor[0][0][0][0])
# Work Well

print(per_channel.int_repr()[0][0][0][0])
per_channel.int_repr()[0][0][0][0]=8
print(per_channel.int_repr()[0][0][0][0])
# Indirectly Assign doesn't work

per_channel[0][0][0][0]
# RuntimeError: Setting strides is possible only on uniformly quantized tensor