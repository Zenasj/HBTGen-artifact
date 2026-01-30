import torch

graph = torch.cuda.CUDAGraph()
a = torch.rand((100,), device="cuda")

try:
    with torch.cuda.graph(graph):
        sampled_indices = torch.multinomial(a, 1, replacement=True)
except Exception as e:
    print(e)

sampled_indices = torch.multinomial(a, 2, replacement=True)

sampled_indices = torch.multinomial(a, 2, replacement=False)