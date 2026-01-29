# torch.rand(seq_len, B, 20, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.sn_rnn = nn.utils.spectral_norm(nn.RNN(20, 10, 1), name='weight_hh_l0')
        
    def forward(self, x):
        out, _ = self.sn_rnn(x)
        return F.log_softmax(out, dim=1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 10, 20, dtype=torch.float32)

