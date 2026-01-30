import torch
from torch.nested import nested_tensor
from torch.nested._nested_tensor import NestedTensor
from torch._inductor import debug

torch._inductor.config.debug = True
torch._dynamo.config.traceable_tensor_subclasses.add(NestedTensor)

device = 'cuda'

def make_tensor(*shape, device=device, dtype=torch.float32):
    return torch.randn(*shape, device=device, dtype=dtype)

torch.manual_seed(1)

def fn(x, x_offsets):
    x_nt = torch._nested_view_from_jagged(x, x_offsets)
    x_nt = x_nt + 69
    x_nt = x_nt * 42
    return x_nt

torch._dynamo.disallow_in_graph(torch.diff)

compiled_fn = torch.compile(fn)

# shape (sum(*), D)
# component shapes: (3, 5), (4, 5), (6, 5)
x = make_tensor(13, 5)
x_offsets = torch.tensor([0, 3, 7, 13], dtype=torch.int64, device=device)

# helps create dynamic graph right away
torch._dynamo.mark_dynamic(x, 0)
torch._dynamo.mark_dynamic(x_offsets, 0)

output = compiled_fn(x, x_offsets)

# shape (sum(*), D)
# component shapes: (2, 5), (6, 5), (4, 5), (5, 5)
y = make_tensor(17, 5)
y_offsets = torch.tensor([0, 2, 8, 12, 17], dtype=torch.int64, device=device)

output2 = compiled_fn(y, y_offsets)

print(output)
print(output2)