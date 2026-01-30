import torch

def test_sparse_sum():
    device = 'cuda'
    i = torch.tensor([[0], [4]], dtype=torch.long, device=device)
    v = torch.tensor([[[-0.4567, -1.8797,  0.0380,  1.4316]]], dtype=torch.double, device=device)
    S = torch.sparse_coo_tensor(i, v, dtype=torch.double, device=device)
    S = S.coalesce()
    S.requires_grad_(True) # when this is True this line cause the memory leak
    print(S)
    f = torch.sparse.sum(S)

test_sparse_sum()
torch.cuda.empty_cache()