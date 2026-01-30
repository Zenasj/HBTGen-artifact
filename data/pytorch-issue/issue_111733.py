import torch

torch._dynamo.reset()
torch._dynamo.config.capture_func_transforms=True

def foo(x):
  return torch.vmap(lambda x: torch.sum(x) + 1e-2)(x)  # Error
  # return torch.vmap(lambda x: torch.mean(x) + 1e-2)(x)  # Error
  # return torch.vmap(lambda x: torch.std(x) + 1e-2)(x)  # Error
  # return torch.vmap(lambda x: torch.sum(x) + torch.tensor(1e-2))(x) # OK
  # return torch.vmap(lambda x: torch.sum(x, 0, keepdim=True) + 1e-2)(x) # OK
  # return torch.vmap(lambda x: torch.square(x) + 1e-2)(x) # OK
  # return torch.vmap(lambda x: x + 1e-2)(x) # OK


torch.compile(foo, fullgraph=True)(torch.randn((3, 3), device='cuda:0'))
# foo(torch.randn((3, 3), device='cuda:0'))   # OK