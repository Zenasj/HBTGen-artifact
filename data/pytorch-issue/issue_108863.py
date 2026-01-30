import torch.nn as nn

import torch
from torch import nn
import copy

class BackboneModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(16, 16, 6)
        self.bn1 = nn.BatchNorm2d(16)
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        return x

model = BackboneModel().eval()
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

old_model = copy.deepcopy(model) # remember the old model
opt_model = torch.compile(model) # compile the model

a = torch.rand(64, 16, 32, 32)

with torch.no_grad():
    model.eval()
    output1 = model(a)
    output2 = opt_model(a)

print("diff of eager model and optimized model at initialization:")
print((output1 - output2).abs().max().item()) # prints 4.172325134277344e-07 on my computer

# train the model and update the weight
model.train()
for i in range(10):
    optim.zero_grad()
    output = opt_model(a)
    output.sum().backward()
    optim.step()

with torch.no_grad():
    model.eval()
    output1 = model(a)
    output2 = opt_model(a)
    output3 = old_model(a)

print("diff of eager model and optimized model after training:")
print((output1 - output2).abs().max().item()) # prints 19307.39453125 on my computer

print("diff of old model and optimized model after training:")
# the opt_model remembers the old model parameters, and don't use the new weight!
print((output3 - output2).abs().max().item()) # prints 4.172325134277344e-07 on my computer

def fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    is_cpu = is_cpu_device(example_inputs)

    fake_mode = detect_fake_mode(example_inputs)

    gm = sink_cat_after_pointwise(gm)
    if config.permute_fusion and not is_cpu:
        # For linear permute fusion, we need to check input info to identify
        # and perform proper permutation/transpose
        ShapeProp(gm, fake_mode=fake_mode).propagate(*example_inputs)
        gm = linear_permute_fusion(gm)
        gm = permute_linear_fusion(gm)
        gm = permute_matmul_fusion(gm)

    # make sure the autograd is disabled.
    if torch.is_grad_enabled():
        return gm
    if not is_cpu:
        return gm
    gm = remove_identity(gm)
    gm = fuse_conv_bn(gm)
    return gm

def conv_bn_computation(x, conv_weight, bn):
    x = conv(x, conv_weight)
    return bn(x)

def fused_conv_bn_computation(x, conv_weight, bn):
    conv_weight = bn(conv_weight) # broadcast the bn operation properly
    x = conv(x, conv_weight)
    return x