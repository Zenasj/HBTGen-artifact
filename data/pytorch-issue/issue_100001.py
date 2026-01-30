import torch.nn as nn

import torch
import torch._dynamo
from torch.nn.modules.lazy import LazyModuleMixin
from functools import partial

class LazyParentModule(LazyModuleMixin, torch.nn.Module):
    def __init__(self):
        super().__init__()

    def initialize_parameters(self, input):
        with torch.no_grad():
            self._val = torch.nn.Parameter(torch.ones(2, 2))

    def forward(self, x):
        return x.sin() + self._val

def test_lazy_module():
    m = LazyParentModule()
    x = torch.rand(2, 2)
    opt_m = torch._dynamo.optimize("eager", nopython=True)(m)
    res = opt_m(x)
    ref = m(x)
    assert torch.allclose(ref, res)

def test_backward_hooks():
    # this test shouldn't care whether hook guards are enabled or not

    class CustomLinear(torch.nn.Module):
        # not an 'allowed module', so should not graph-break
        def __init__(self, a, b):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(a, b))

        def forward(self, x):
            return torch.mm(x, self.weight)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                *[CustomLinear(10, 10)]
                + [CustomLinear(10, 10000)]
                + [CustomLinear(10000, 5)]
            )

        def forward(self, x):
            return self.net(x)

    model = ToyModel()
    backward_hook_handles = {}
    pre_backward_hook_handles = {}

    grad_sizes = {}

    def backward_hook(name, mod, grad_inp, grad_out):
        grad_sizes[name] = (
            (gi.shape for gi in grad_inp),
            (go.shape for go in grad_out),
        )
        return None

    pre_grad_sizes = {}

    def backward_pre_hook(name, mod, grad_out):
        pre_grad_sizes[name] = (go.shape for go in grad_out)
        return None

    for name, module in model.named_modules():
        backward_hook_handles[name] = module.register_full_backward_hook(
            partial(backward_hook, name)
        )

        pre_backward_hook_handles[name] = module.register_full_backward_pre_hook(
            partial(backward_pre_hook, name)
        )

    model = torch.compile(model, backend="aot_eager")

    for i in range(2):
        # second iteration is key, hooks would have fired during aot trace
        # on first iter
        x = torch.randn((20, 10))
        pred = model(x)
        loss = pred.sum()
        loss.backward()

    assert (grad_sizes.keys() == backward_hook_handles.keys())
    assert (pre_grad_sizes.keys() == pre_backward_hook_handles.keys())

test_lazy_module()
test_backward_hooks()