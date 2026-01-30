import torch
device = 'cuda'
nnz = 6
N = 1000000
size = (N,) * 4

indices = torch.randint(N, (4, nnz))
values = torch.rand(nnz)

try:
    s1 = torch.sparse_coo_tensor(indices, values, size, device=device).coalesce()
except RuntimeError as msg:
    # assuming PR 57492
    print(f'failed to coalesce: {msg}')
else:
    print(s1)
    print("increase N to reproduce gh-57416")
    
ordering = torch.tensor(sorted(range(nnz), key=lambda i: tuple(indices[:, i])))
sorted_indices = torch.index_select(indices, 1, ordering)
sorted_values = torch.index_select(values, 0, ordering)
s2 = torch.sparse_coo_tensor(sorted_indices, sorted_values, size, device=device)._coalesced_(True)
print(s2)