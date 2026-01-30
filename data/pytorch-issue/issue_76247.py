import torch

with torch.cuda.graph(torch.cuda.CUDAGraph()):
    torch.zeros(2 ** 40, device="cuda")

import pytest
import torch

with pytest.raises(RuntimeError, match="CUDA out of memory"):
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        torch.zeros(2 ** 40, device="cuda")