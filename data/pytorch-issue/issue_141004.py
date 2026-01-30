import torch

data0 = [
    -9.2234e+10,  1.6106e+09, -1.1529e+18, -4.9844e+08, -3.6029e+16,
    -1.6106e+09,  5.3687e+08, -4.1943e+06,  2.5600e+02, -1.0000e+00
]
data1 = [
    -8.6085e+18,  2.1475e+09, -4.9844e+08,  8.9681e+18, -1.2510e+12,
     1.8790e+09,  2.1475e+09,  3.0000e+00, -1.0128e+09, -1.6762e+18
]
crow_indices = torch.tensor(data0, dtype=torch.long)
col_indices = torch.tensor(data1, dtype=torch.long)
out_int32 = True
transpose = False
torch.ops.aten._convert_indices_from_csr_to_coo(crow_indices, col_indices, out_int32=out_int32, transpose=transpose)

import torch

crow_indices = torch.tensor([
    -9.2234e+10,  1.6106e+09, -1.1529e+18, -4.9844e+08, -3.6029e+16,
    -1.6106e+09,  5.3687e+08, -4.1943e+06,  2.5600e+02, -1.0000e+00
], dtype=torch.long)
col_indices = torch.tensor([
    -8.6085e+18,  2.1475e+09, -4.9844e+08,  8.9681e+18, -1.2510e+12,
     1.8790e+09,  2.1475e+09,  3.0000e+00, -1.0128e+09, -1.6762e+18
], dtype=torch.long)
values = torch.arange(1, len(crow_indices) + 1, dtype=torch.float)
rows = 10
cols = 10

sparse_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(rows, cols))
sparse_tensor.to_sparse_coo()

import torch
with torch.sparse.check_sparse_tensor_invariants():
    crow_indices = torch.tensor([
        -9.2234e+10,  1.6106e+09, -1.1529e+18, -4.9844e+08, -3.6029e+16,
        -1.6106e+09,  5.3687e+08, -4.1943e+06,  2.5600e+02, -1.0000e+00
    ], dtype=torch.long)
    col_indices = torch.tensor([
        -8.6085e+18,  2.1475e+09, -4.9844e+08,  8.9681e+18, -1.2510e+12,
         1.8790e+09,  2.1475e+09,  3.0000e+00, -1.0128e+09, -1.6762e+18
    ], dtype=torch.long)
    values = torch.arange(1, len(crow_indices) + 1, dtype=torch.float)
    rows = 10
    cols = 10

    sparse_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(rows, cols))
    sparse_tensor.to_sparse_coo()

import torch

crow_indices = torch.full((10,), -9.2234e+10, dtype=torch.long)
col_indices = torch.full((10,), -8.6085e+18, dtype=torch.long)
out_int32 = True
transpose = False

output = torch.ops.aten._convert_indices_from_csr_to_coo(crow_indices, col_indices, out_int32=out_int32, transpose=transpose)
print(output)