import torch

def reprod_function(t: torch.Tensor, a: torch.Tensor):
    t[a] *= 0.2
    return t

reprod_function_compile = torch.compile(reprod_function, backend="eager", fullgraph=True)

x  = torch.rand(1,2,3)
y = torch.rand(1,2,3).to(torch.bool)
z = reprod_function(x,y)
print(z.dtype)
z = reprod_function_compile(x,y)
print(z.dtype)

t = torch.where(a, t * 0.2, t)