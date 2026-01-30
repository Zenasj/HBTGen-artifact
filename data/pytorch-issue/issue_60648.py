import torch

for dtype in [torch.bool, torch.half, torch.bfloat16, torch.complex64, torch.complex128]:
    a = torch.ones(3, 3, dtype=dtype)
    a_csr = a.to_sparse_csr()
    try:
        a_csr.to_dense()
    except RuntimeError as e:
        print(e)