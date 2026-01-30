import torch.nn as nn

3
import torch

n, c, hi, wi = 1025, 2, 256, 256
ho, wo = 1024, 1024
input = torch.randn(n, c, hi, wi, dtype=torch.float32).cuda()
input.requires_grad_(True)
output_size = [ho, wo]
output = torch.nn.functional.interpolate(input, size=output_size, mode ='nearest')
grad_output = torch.randn(n, c, ho, wo, dtype=torch.float32).cuda()
output.backward(grad_output)
grad_input = input.grad
print(grad_input.cpu())