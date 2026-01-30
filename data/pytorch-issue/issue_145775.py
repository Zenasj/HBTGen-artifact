import torch

torch._dynamo.config.automatic_dynamic_local_pgo = False

@torch.compile()
def fn(x):
    return torch.cat([x, torch.ones(5, 5)])

x = torch.ones(5, 5)
torch._dynamo.decorators.mark_unbacked(x, 0)
torch._dynamo.decorators.mark_unbacked(x, 1)
fn(x)

L_x_: "f32[u0, 5][5, 1]cpu"