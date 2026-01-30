import torch

def test(x: torch.Tensor, y: torch.Tensor):
    return x, y * 2

vmap_test = torch.vmap(test, in_dims=(None, 0), out_dims=(None, 0))
compiled_vmap_test = torch.compile(vmap_test, fullgraph=True)
print(compiled_vmap_test(torch.rand(3), torch.rand(3, 4)))