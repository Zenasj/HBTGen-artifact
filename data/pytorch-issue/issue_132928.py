import torch.nn as nn

import torch
from torch.sparse import to_sparse_semi_structured

def run_sparse_matmul(M, N, K, force_cutlass=False, dtype=torch.float16, device='cuda'):
    torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = force_cutlass

    # Create sparse semi-structured matrix
    x = torch.randn(M, N, dtype=dtype, device=device)
    dense_weights = torch.randn(K, N, dtype=dtype, device=device)
    sparse_weights = to_sparse_semi_structured(dense_weights)

    torch.nn.functional.linear(x, sparse_weights)

# Fails with:
# sparse_semi_structured_mad_op : Supported only on GPUs with compute capability 8.x
run_sparse_matmul(64, 64, 64, True)

# Fails with:
# CUDA error: architecture mismatch when calling `cusparseLtInit(&handle)`
run_sparse_matmul(64, 64, 64, False)