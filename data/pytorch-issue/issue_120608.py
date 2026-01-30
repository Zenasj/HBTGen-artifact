import torch
import torch._dynamo as dynamo
from torch._functorch.aot_autograd import aot_module_simplified

dynamo.reset()

def raw_function(t):
    out = torch.prod(t, 3, keepdim=True)
    return out

input_shapes = [(8, 10, 3, 2), (8, 3, 5, 2), (8, 4, 8, 2)]

compiled_fn = torch.compile(raw_function, backend="inductor")

for s in input_shapes:
    t1 = torch.randn(s, requires_grad=True)
    t1_h = t1.to("cpu")
    h_result = compiled_fn(t1_h)
    grad = torch.ones_like(h_result)
    h_result.backward(grad)