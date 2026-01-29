# torch.rand(50, 16, 128, dtype=torch.float32).cuda()
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 50).cuda()  # latent_dim=128, vocab_size=50

    def forward(self, x):
        return self.linear(x).log_softmax(-1)

def my_model_function():
    return MyModel()

def GetInput():
    seq_len = 50
    batch_size = 16
    latent_dim = 128
    x = torch.randn(seq_len, batch_size, latent_dim, dtype=torch.float32).cuda()
    return x

