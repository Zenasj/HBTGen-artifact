# torch.rand(20, 32, 100, dtype=torch.float32)  # (seq_len, batch_size, input_size)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.gru = nn.GRU(100, 20, bias=False)
        # Apply weight normalization to both input and hidden weights
        self.gru = nn.utils.weight_norm(self.gru, name='weight_ih_l0')
        self.gru = nn.utils.weight_norm(self.gru, name='weight_hh_l0')

    def forward(self, x):
        return self.gru(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(20, 32, 100, dtype=torch.float32).cuda()

