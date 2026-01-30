import torch
import torch.nn as nn

class GraphModule(torch.nn.Module):
    def forward(self, x : torch.Tensor):
        # No stacktrace found for following nodes
        _set_grad_enabled = torch._C._set_grad_enabled(False)

        # File: tmp3.py:7, code: x.add_(1)
        add_ = x.add_(1);  x = None

        # No stacktrace found for following nodes
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
        return ()

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[2]):
        # No stacktrace found for following nodes
        clone: f32[2] = torch.ops.aten.clone.default(primals_1);  primals_1 = None

        # File: tmp3.py:7, code: x.add_(1)
        add: f32[2] = torch.ops.aten.add.Tensor(clone, 1);  clone = None
        return [add]

# create a non-leaf
a = torch.ones(2, requires_grad=True).clone()
# mutate it in no_grad() mode
with torch.no_grad():
    a.mul_(2)