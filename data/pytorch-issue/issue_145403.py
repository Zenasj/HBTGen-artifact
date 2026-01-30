import torch

torch._dynamo.config.automatic_dynamic_local_pgo = False

@torch.compile()
def fn(x):
    y = torch.cat([x, x])
    torch._dynamo.graph_break()
    z = torch.cat([y, y])
    torch._dynamo.graph_break()
    return torch.cat([z, z])

x = torch.ones(5, 5)
torch._dynamo.decorators.mark_unbacked(x, 0)
torch._dynamo.decorators.mark_unbacked(x, 1)
fn(x)