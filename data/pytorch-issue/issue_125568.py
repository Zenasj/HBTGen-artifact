import torch

def f(a, tmp):
    a_view = a.view(-1)
    torch._dynamo.graph_break()
    with torch.no_grad():
        a.set_(tmp)
        a_view.mul_(2)
    return a + tmp

inp = torch.ones(3, 3, requires_grad=True)
tmp = torch.ones(3, 3, requires_grad=True)

opt_f = torch.compile(f, backend="eager")
opt_f(inp, tmp)
print(torch.is_grad_enabled())