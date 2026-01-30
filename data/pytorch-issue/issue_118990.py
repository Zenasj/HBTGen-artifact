import torch
import torch.nn as nn

linear_in = 1 
linear_out = 2 # linear_out != 1 
bn_in = 5 # bn_in != linear_out

# model
linear = nn.Linear(linear_in, linear_out)
bn = nn.BatchNorm2d(bn_in)
bn.running_mean = torch.rand(bn_in)
bn.running_var = torch.rand(bn_in)
linear.eval()
bn.eval()
# data
data = torch.randn(1, bn_in, 1, linear_in)

# eager
print("bn(linear(data)): ", bn(linear(data)))

# manual linear-bn folding
weight = linear.weight.view(1, -1).repeat(bn_in, 1).view(1, bn_in, 1, -1)
bias = linear.bias.view(1, -1).repeat(bn_in, 1).view(1, bn_in, 1, -1)
bn_scale = torch.rsqrt(bn.running_var + bn.eps).reshape(1, -1, 1, 1)
fused_weight = weight * bn_scale
fused_bias = (bias - bn.running_mean.reshape(1, -1, 1, 1)) * bn_scale + bn.bias.reshape(1, -1, 1, 1)
print("fused_weight: ", fused_weight.shape) # (1, bn_in, 1, linear_in)
print("fused_bias: ", fused_bias.shape) # (1, bn_in, 1, linear_in)
print("fused_weight * data + fused_bias: ", fused_weight * data + fused_bias)
print(torch.allclose(bn(linear(data)), fused_weight * data + fused_bias))

# linear-bn folding
nn.utils.fusion.fuse_linear_bn_eval(linear, bn)