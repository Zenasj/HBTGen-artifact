import torch.nn as nn

import torch
class TensorConstant(torch.nn.Module):
    def __init__(self, value):
        # type: (Tensor) -> None
        super(TensorConstant, self).__init__()
        self.val = torch.tensor(value)

    def forward(self, xs):
        # type: (Tuple[Dict[str,Tensor], Dict[str, Tensor]]) -> Tensor
        reals, bools = xs
        return self.val

class Real(torch.nn.Module):
    def __init__(self, name):
        super(Real, self).__init__()
        self.name = name

    def forward(self, xs):
        # type: (Tuple[Dict[str,Tensor], Dict[str, Tensor]]) -> Tensor
        reals, bools = xs
        return reals[self.name]

class Minus(torch.nn.Module):
    def __init__(self, f1, f2):
        super(Minus, self).__init__()
        self.f1 = f1
        self.f2 = f2

    def forward(self, xs):
        # type: (Tuple[Dict[str,Tensor], Dict[str, Tensor]]) -> Tensor
        val = self.f1.forward(xs) - self.f2.forward(xs)
        return val

xs = ({'foo': torch.tensor(1.0, requires_grad=True)}, {})
f1 = TensorConstant(2.0)
f2 = Real("foo")
model = Minus(f1, f2)

print(model(xs)) # works

scripted_model = torch.jit.script(model)
print(scripted_model(xs)) # fails

xs = ({'foo': torch.tensor(1.0, requires_grad=True)}, {})
f1 = TensorConstant(2.0)
f2 = Real("foo")
model = Minus(f1, f2)

print(model(xs)) # works
# tensor(1., grad_fn=<SubBackward0>)

scripted_model = torch.jit.script(model)
print(scripted_model(xs)) # fails
# tensor(1., grad_fn=<SubBackward0>)