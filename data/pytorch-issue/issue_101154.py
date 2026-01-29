# torch.rand(5, 5, dtype=torch.float32)
import torch
import sys
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(5, 5)
        self.dropout = nn.Dropout()

    def _init_attention_seed(self):
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            self.attention_seed = int(torch.seed() % sys.maxsize)
        torch.manual_seed(self.attention_seed)

    def forward(self, x):
        self._init_attention_seed()
        return self.dropout(self.linear(x))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(5, 5)

