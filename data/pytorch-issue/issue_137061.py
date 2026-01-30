import torch

def sum_of_squares(x):
    return torch.sum(x ** 2)

@torch.compile(backend="eager", fullgraph=True, dynamic=True)
def fn(batch_size, vector_length, func):
    x = torch.randn(batch_size, vector_length)
    vmapped_func = torch.vmap(func)
    return vmapped_func(x)

batch_size = 5
vector_length = 4
print(fn(batch_size, vector_length, sum_of_squares))

# Re-compile here as we add constant match guard to

dynamic=True

batch_size

in_dims