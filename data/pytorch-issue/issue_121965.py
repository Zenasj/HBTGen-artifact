import torch


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")


# Complex tensors
src = torch.tensor([1 + 2j, 3 - 4j, 1 + 1j], dtype=torch.cfloat, device=device)
indices = torch.tensor([0, 0, 0], device=device)

# src = torch.tensor([1 + 2j, 3 - 4j, 1 + 1j], dtype=torch.complex64)
# indices = torch.tensor([0, 0, 0])


output = src.scatter_reduce(dim=0, index=indices, src=src, reduce='prod', include_self=False)
print(output)