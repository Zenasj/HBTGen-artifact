import torch
import torch.nn as nn
import math

# torch.rand(B, 16, dtype=torch.float32, device='cuda')  # Input shape (batch, 16)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 16, bias=False)
        self.linear2 = nn.Linear(16, 32, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))

def my_model_function():
    model = MyModel()
    model.cuda()  # Matches original setup in the issue
    return model

def newtonschulz5(G, steps: int, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

@torch.compile
def scaled_newton_schulz(G, steps: int):
    shape = G.shape
    dtype = G.dtype
    G = G.reshape(shape[0], -1)
    G = newtonschulz5(G, steps)
    G = G.reshape(shape).type(dtype)
    G = G * math.sqrt(max(1, shape[0] / G[0].numel()))
    return G

def GetInput():
    return torch.randn(4, 16, device="cuda")  # Matches original input dimensions

