import torch.nn as nn

import torch

from functorch.experimental.control_flow import cond


class MySubModule(torch.nn.Module):
    def foo(self, x):
        return x.cos()

    def forward(self, x):
        return self.foo(x)


class CondBranchClassMethod(torch.nn.Module):
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
      - both branches must take the same args, which must also match the branch args passed to cond.
      - both branches must return a single tensor
      - returned tensor must have the same tensor metadata, e.g. shape and dtype
      - branch function can be free function, nested function, lambda, class methods
      - branch function can not have closure variables
      - no inplace mutations on inputs or global variables


    This example demonstrates using class method in cond().

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    def __init__(self):
        super().__init__()
        self.subm = MySubModule()

    def bar(self, x):
        return x.sin()

    def forward(self, x):
        return cond(x.shape[0] <= 2, self.subm.forward, self.bar, [x])

model = CondBranchClassMethod()
input = torch.randn(5)

# exported_program = torch.export.export(model, args=(input,))(input)  # works
onnx_program = torch.onnx.dynamo_export(model, input)  # Unknown call_function target: cond

import torch


class MySubModule(torch.nn.Module):
    def foo(self, x):
        return x.cos()

    def forward(self, x):
        return self.foo(x)


class CondBranchClassMethod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.subm = MySubModule()

    def bar(self, x):
        return x.sin()

    def forward(self, x):
        return torch.cond(x.size(0) <= 2, self.subm, self.bar, [x])

model = CondBranchClassMethod()
input = torch.randn(5)

onnx_program = torch.onnx.export(model, (input,), dynamo=True, dynamic_shapes={'x': {0: torch.export.Dim("batch_size")}})
print(onnx_program)