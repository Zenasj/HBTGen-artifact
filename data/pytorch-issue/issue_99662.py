import torch.nn as nn
import torchvision

import torch
from torchvision.models import resnet18

torch.onnx.dynamo_export(resnet18(), torch.randn(1, 3, 224, 224))

import torch
import torch._dynamo
from torch import func
from torch.fx.experimental import proxy_tensor
from torchvision.models import resnet18

dummy_input = torch.randn(1, 3, 224, 224)
gm, _ = torch._dynamo.export(resnet18(), dummy_input, aten_graph=True)
gm = proxy_tensor.make_fx(func.functionalize(gm))(dummy_input)

"""
Traceback (most recent call last):
  File "repro_resnet.py", line 9, in <module>
    gm = proxy_tensor.make_fx(func.functionalize(gm))(dummy_input)
  File "/home/bowbao/pytorch/torch/fx/experimental/proxy_tensor.py", line 771, in wrapped
    t = dispatch_trace(wrap_key(func, args, fx_tracer, pre_autograd), tracer=fx_tracer, concrete_args=tuple(phs))
  File "/home/bowbao/pytorch/torch/_dynamo/eval_frame.py", line 252, in _fn
    return fn(*args, **kwargs)
  File "/home/bowbao/pytorch/torch/fx/experimental/proxy_tensor.py", line 467, in dispatch_trace
    graph = tracer.trace(root, concrete_args)
  File "/home/bowbao/pytorch/torch/_dynamo/eval_frame.py", line 252, in _fn
    return fn(*args, **kwargs)
  File "/home/bowbao/pytorch/torch/fx/_symbolic_trace.py", line 778, in trace
    (self.create_arg(fn(*args)),),
  File "/home/bowbao/pytorch/torch/fx/experimental/proxy_tensor.py", line 484, in wrapped
    out = f(*tensors)
  File "<string>", line 1, in <lambda>
  File "/home/bowbao/pytorch/torch/_functorch/vmap.py", line 39, in fn
    return f(*args, **kwargs)
  File "/home/bowbao/pytorch/torch/_functorch/eager_transforms.py", line 1600, in wrapped
    func_outputs = func(*func_args, **func_kwargs)
  File "/home/bowbao/pytorch/torch/fx/graph_module.py", line 662, in call_wrapped
    return self._wrapped_call(self, *args, **kwargs)
  File "/home/bowbao/pytorch/torch/fx/graph_module.py", line 281, in __call__
    raise e
  File "/home/bowbao/pytorch/torch/fx/graph_module.py", line 271, in __call__
    return super(self.cls, obj).__call__(*args, **kwargs)  # type: ignore[misc]
  File "/home/bowbao/pytorch/torch/fx/_symbolic_trace.py", line 756, in module_call_wrapper
    return self.call_module(mod, forward, args, kwargs)
  File "/home/bowbao/pytorch/torch/fx/experimental/proxy_tensor.py", line 433, in call_module
    return forward(*args, **kwargs)
  File "/home/bowbao/pytorch/torch/fx/_symbolic_trace.py", line 749, in forward
    return _orig_module_call(mod, *args, **kwargs)
  File "/home/bowbao/pytorch/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "<eval_with_key>.4", line 15, in forward
  File "/home/bowbao/pytorch/torch/_ops.py", line 398, in __call__
    return self._op(*args, **kwargs or {})
RuntimeError: false INTERNAL ASSERT FAILED at "/home/bowbao/pytorch/build/aten/src/ATen/RegisterFunctionalization_2.cpp":7718, please report a bug to PyTorch. mutating a non-functional tensor with a functional tensor is not allowed. Please ensure that all of your inputs are wrapped inside of a functionalize() call.
"""

import torch
from torch.fx.experimental import proxy_tensor
from typing import Callable
from torch.utils import _pytree as pytree


def _functionalize(function: Callable) -> Callable:
    def wrapped(*inputs):
        inputs_functional = pytree.tree_map_only(
            torch.Tensor, torch._to_functional_tensor, inputs
        )
        torch._enable_functionalization(reapply_views=True)
        try:
            out = function(*inputs_functional)
        finally:
            torch._disable_functionalization()
        flat_inputs, _ = pytree.tree_flatten(inputs)
        flat_inputs_functional, _ = pytree.tree_flatten(inputs_functional)
        for inpt, input_functional in zip(flat_inputs, flat_inputs_functional):
            if isinstance(input_functional, torch.Tensor):
                torch._sync(input_functional)
                inpt_new = torch._from_functional_tensor(input_functional)
        pytree.tree_map(torch._sync, out)
        out_unwrapped = pytree.tree_map(torch._from_functional_tensor, out)
        return out_unwrapped

    return wrapped


dummy_input = torch.randn(2, 3, 224, 224)


class BNModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        return self.bn(x)


gm = proxy_tensor.make_fx(
    _functionalize(BNModule()), tracing_mode="symbolic", _allow_non_fake_inputs=True
)(dummy_input)
# RuntimeError: false INTERNAL ASSERT FAILED at "/home/bowbao/pytorch_dev/build/aten/src/ATen/RegisterFunctionalization_2.cpp":7731, 
# please report a bug to PyTorch. mutating a non-functional tensor with a functional tensor 
# is not allowed. Please ensure that all of your inputs are wrapped inside of a functionalize() 
# call.

from torch._functorch.aot_autograd import aot_function
def print_compile_fn(fx_module, args):
    print(fx_module)
    return fx_module
aot_fn = aot_function(BNModule(), print_compile_fn)
aot_fn(dummy_input)