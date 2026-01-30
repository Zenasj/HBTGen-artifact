import torch

self = torch.full((3, 4, 5,), 1, dtype=torch.float32, requires_grad=False).to_mkldnn()
dim0 = 1250999896764
dim1 = 0
torch._mkldnn_transpose(self, dim0, dim1)