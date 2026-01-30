from typing import Tuple
import torch

size = (int(3e4), int(3e4))

# utils to create sample tensor
def create_sparse_tensor(sparse_rate: float, size: Tuple[int, int], device="cuda"): 
    num_values = int(size[0] * size[1] * (1 - sparse_rate))
    v = torch.ones(num_values, device=device)
    ind = torch.stack([
        torch.randint(size[0], size=(num_values,), device=device), 
        torch.randint(size[1], size=(num_values,), device=device)
    ])
    return torch.sparse_coo_tensor(
        ind,
        v,
        size=size,
    )

tensor = create_sparse_tensor(0.9, size, device="cpu")
x = torch.randint(size[0], size=(39,))

tensor = tensor.to("cuda")
x = x.to("cuda")

print(torch.cuda.max_memory_allocated()) # 1801454080

tensor.to_dense().index_select(0, x);
print(torch.cuda.max_memory_allocated())  # 6122264064

tensor.index_select(0, x);
print(torch.cuda.max_memory_allocated()) # 6154191872