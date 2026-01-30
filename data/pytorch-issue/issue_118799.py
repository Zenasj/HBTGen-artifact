import torch
from torch.testing._internal.two_tensor import TwoTensor


@torch._dynamo.allow_in_graph
def to_subclass(x):
    return TwoTensor(x.clone(), x.clone())

@torch.compile(backend="aot_eager")
def f(x):
    tmp_subclass = to_subclass(x)
    return tmp_subclass.view(-1)

x = torch.ones(2)
out = f(x)
print(type(out.a))
print(type(out.b))

def forward(self, arg0_1: "f32[2]"):
        # No stacktrace found for following nodes
        _tensor_constant0 = self._tensor_constant0
        return (_tensor_constant0,)