import torch
import torch.nn as nn
import torch.nn.functional as F

a_cpu = torch.Tensor(1,1,4,4).uniform_()
a_gpu = a_cpu.cuda()

a_cpu.requires_grad_();
a_gpu.requires_grad_();

scale_factor = 0.25
def upsample(t):
    return F.interpolate(t, scale_factor=scale_factor, mode='nearest')


upsampled_cpu = upsample(a_cpu)
upsampled_gpu = upsample(a_gpu)

grad_cpu = upsampled_cpu.clone().uniform_()
grad_gpu = grad_cpu.cuda()

upsampled_cpu.backward(grad_cpu)
upsampled_gpu.backward(grad_gpu)

input_grad_cpu = a_cpu.grad
input_grad_gpu = a_gpu.grad

print('output match:', upsampled_cpu.allclose(upsampled_gpu.cpu()))
print('gradinput match:', input_grad_cpu.allclose(input_grad_gpu.cpu()))
print('difference:')
print(input_grad_cpu - input_grad_gpu.cpu())

# out = F.interpolate(input_tensor, scale_factor=(scale, scale))
inv_scale = 1/scale
int_inv_scale = int(inv_scale)
assert(inv_scale == int_inv_scale)
out = input_tensor[:, :, ::int_inv_scale, ::int_inv_scale]