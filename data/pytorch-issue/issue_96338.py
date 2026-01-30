import torch

y = torch.randn([3, 3, 3])
def my_dyn_fn(x):
    if x.shape[0] == 3:
        return x.sin()
    return x.cos()

graph, guards = torch._dynamo.export(my_dyn_fn, y, constraints=[dynamic_dim(y, 0)])