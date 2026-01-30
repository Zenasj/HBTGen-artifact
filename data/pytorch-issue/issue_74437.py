import torch
torch._C._jit_get_operation("aten::cudnn_convolution_backward_weight")

empty_weight = torch.empty(weight_shape, dtype=input.dtype, layout=input.layout, device=input.device)

empty_weight = torch.tensor(0.0, dtype=input.dtype, layout=input.layout, device=input.device).expand(weight_shape)