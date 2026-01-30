import torch.nn as nn

import torch

class Module(torch.nn.Module):
    def __init__(self, y):
        super().__init__()
        self.y = y

    @torch._dynamo.assume_constant_result
    def check(self):
        return self.y.item() == 1

    def forward(self, x):
        # This line leads to module obj being tracked as UnspecializedNNModuleVariable in dynamo
        # Commenting out this line will make it pass.
        self.device = x.device

        if self.check():
            return x + 1
        else:
            return x + 2

model = Module(torch.tensor([1]))
x = torch.tensor(1)
model(x)
# torch._dynamo.exc.InternalTorchDynamoError: Module.check() missing 1 required positional argument: 'self'
torch.compile(model)(x)

import torch

@torch._dynamo.assume_constant_result
def outer_check(y):
    return y.item() == 1

class Module(torch.nn.Module):
    def __init__(self, y):
        super().__init__()
        self.y = y

    def forward(self, x):
        self.device = x.device

        if outer_check(self.y):
            return x + 1
        else:
            return x + 2

model = Module(torch.tensor([1]))
x = torch.tensor(1)
model(x)
"""
  File "/home/bowenbao/anaconda3/envs/torch/lib/python3.11/site-packages/torch/_subclasses/fake_impls.py", line 284, in local_scalar_dense
    raise DataDependentOutputException(func)
torch._dynamo.exc.InternalTorchDynamoError: aten._local_scalar_dense.default
"""
torch.compile(model)(x)