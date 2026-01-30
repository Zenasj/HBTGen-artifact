import torch

def foo(x):
    a = x.item()
    torch._constrain_as_size(a, min=1, max=10)
    return torch.ones(a, a)

dynamo_config.capture_scalar_outputs = True 
fn = torch.compile(foo, fullgraph=True, dynamic=True)
fn(torch.tensor(5))