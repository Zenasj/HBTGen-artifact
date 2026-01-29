import torch
import torch.nn as nn
import cupy
from torch.utils.dlpack import from_dlpack, to_dlpack

# torch.randint(B, L, high=131072, dtype=torch.long)  # Example input shape (10, 5)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(131072, 512, sparse=True).cuda()
        initrange = 0.5 / self.embedding.embedding_dim
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # Generate CuPy array and replace weights using DLPack
        sampl = cupy.random.uniform(low=-1, high=1, size=self.embedding.weight.shape)
        dlpack_tensor = from_dlpack(sampl.toDlpack())
        self.embedding.weight.data = dlpack_tensor.cuda()  # Ensure tensor is on CUDA

    def forward(self, indices):
        return self.embedding(indices)

def my_model_function():
    return MyModel()

def GetInput():
    import torch
    indices = torch.randint(0, 131072, (10, 5), dtype=torch.long, device='cuda')
    return indices

