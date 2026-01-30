import torch
import torch.nn as nn

bn = torch.nn.BatchNorm2d(4).to('cuda')
bn.train()
x = torch.ones((1,4, 20, 20)).to('cuda')
x[0, 1, 3, 3] = float('inf')
with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
    y = bn(x)
state_dict = bn.state_dict()
print(state_dict["running_mean"])  # => tensor([0.1000,    inf, 0.1000, 0.1000], device='cuda:0')
print(state_dict["running_var"])  # => tensor([0.9000,    nan, 0.9000, 0.9000], device='cuda:0')

def nan_hook(name):
    def hook(m, input, output):
        if not torch.isfinite(output).all():
            print("Invalid output in {}".format(name))
    return hook

for name, module in model.named_modules():
    module.register_forward_hook(nan_hook(name))