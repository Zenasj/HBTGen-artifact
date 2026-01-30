import torch.nn as nn
import numpy as np

import torch

def solve(x, other, out):
    torch.linalg.solve(x, other, out=out)

if __name__ == "__main__":
    dense_inp = torch.randn((1000, 1000), dtype=torch.float64)
    # Set 50% of the values to 0 randomly
    dense_inp = torch.nn.functional.dropout(dense_inp, p=0.5)
    sparse_inp = dense_inp.to_sparse_csr()

    other = torch.randint(100, (1000,), dtype=torch.float64)
    out = torch.randint(1, (1000,), dtype=torch.float64)

    solve(sparse_inp, other, out)