import torch
import torch._dynamo

def fn(x):
    y = x.permute(0, 2, 3, 1).contiguous()
    torch._dynamo.graph_break()
    return y.view(-1, 4)

opt_fn = torch._dynamo.optimize("inductor")(fn)
x = torch.rand([4, 4, 4, 4])
print(opt_fn(x))

clone

memory_format

contiguous

aten.clone

memory_format

clone