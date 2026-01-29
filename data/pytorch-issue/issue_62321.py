# torch.rand(B, 10, 512, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=3)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        output = self.transformer_encoder(x)
        i = 0
        while i < 10:
            if i is 5:  # This line introduces the problematic 'is' operator (__is_)
                break
            i += 1
        return self.fc(output)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 10, 512, dtype=torch.float32)

