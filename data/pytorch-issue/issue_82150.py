import torch
dtypes = [torch.bool, torch.int8, torch.int32, torch.bfloat16, torch.float32, torch.float64]
for dtype in dtypes:
    a = torch.ones((6, 1), dtype=dtype)
    a_s = a.to_sparse_coo()
    if a_s[0].dtype != dtype:
        print(f'2d->1d index select (with nnz != 0) result dtype incorrect, for dtype {dtype}, got {a_s[0].dtype}')
    if a_s[0,0].dtype != dtype:
        print(f'2d->scalar index select (with nnz != 0) result dtype incorrect, for dtype {dtype}, got {a_s[0,0].dtype}')

    b_s = (a * 0).to_sparse_coo()
    if b_s[0].dtype != dtype:
        print(f'2d->1d index select (with nnz == 0) result dtype incorrect, for dtype {dtype}, got {b_s[0].dtype}')
    if b_s[0,0].dtype != dtype:
        print(f'2d->scalar index select (with nnz == 0) result dtype incorrect, for dtype {dtype}, got {b_s[0,0].dtype}')