import torch

torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_scalar_outputs = True

nt = torch.nested.nested_tensor([
    torch.randn(2),
    torch.randn(3),
    torch.randn(4),
], layout=torch.jagged, device="cuda")

@torch.compile(fullgraph=True)
def f(t, mask):
    nt = torch.nested.masked_select(t, mask)
    return torch.where(nt > 0., torch.ones_like(nt), torch.zeros_like(nt))

t = torch.randn(3, 5)
mask = torch.randint(0, 2, t.shape, dtype=torch.bool)
output = f(t, mask)