import torch 
import torch.nn as nn

batch = 1
in_features = 4
out_features = 8
device = f'cpu'
factory_kwargs = {'device': device, 'dtype': torch.float32}


def hook_fn(module, grad_input, grad_output):
    params = list(module.parameters())
    for param in params:
        print(f"hook grad: {param.grad}")


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias=False, **factory_kwargs)

    def forward(self, x):
        x = self.fc(x)
        return x

model = Model()
model.register_full_backward_hook(hook_fn)

input = torch.randn(batch, in_features, **factory_kwargs)


output = model(input)
output.backward(torch.randn_like(output))

for param in model.parameters():
    print(f"grad: {param.grad}")

import torch 
import torch.nn as nn

batch = 1
in_features = 4
out_features = 8
device = f'cpu'
factory_kwargs = {'device': device, 'dtype': torch.float32}


def hook_fn(module, grad_input, grad_output):
    params = list(module.parameters())
    for param in params:
        print(f"hook grad: {param.grad}")


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias=False, **factory_kwargs)

    def forward(self, x):
        x = self.fc(x)
        return x

model = Model()
model.register_full_backward_hook(hook_fn)

input = torch.randn(batch, in_features, **factory_kwargs, requires_grad=True)

output = model(input)
output.backward(torch.randn_like(output))

for param in model.parameters():
    print(f"grad: {param.grad}")