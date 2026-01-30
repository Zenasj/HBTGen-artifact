import torch.nn as nn

import torch 
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

device = "cuda"
SparseSemiStructuredTensor._FORCE_CUTLASS = True

input = torch.rand(64, 768, 768, device=device).half()
model = (
    nn.Sequential(
        nn.Linear(768, 3072),
        nn.Linear(3072, 768),
    )
    .half()
    .to(device)
)

for i in range(2):
    m, n = model[i].weight.shape
    mask = torch.Tensor([0, 0, 1, 1]).tile(m, n // 4).to(device).bool()
    # set masked weight
    model[i].weight = nn.Parameter(model[i].weight * mask)

dense_result = model(input)

for i in range(2):
    model[i].weight = nn.Parameter(to_sparse_semi_structured(model[i].weight))

sparse_result = model(input)

assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)