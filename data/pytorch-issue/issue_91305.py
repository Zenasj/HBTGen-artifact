# torch.rand(batch_size, max_seqlen, 768, dtype=...)  # Inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.vanilla_layers = nn.ModuleList()
        self.bt_layers = nn.ModuleList()
        for _ in range(12):
            vanilla_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12)
            bt_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12)
            vanilla_layer.norm2.eps = 2e-5  # disable fastpath
            self.vanilla_layers.append(vanilla_layer)
            self.bt_layers.append(bt_layer)

    def forward(self, inputs):
        vanilla_output = inputs
        bt_output = inputs
        for vanilla_layer, bt_layer in zip(self.vanilla_layers, self.bt_layers):
            vanilla_output = vanilla_layer(vanilla_output)
            bt_output = bt_layer(bt_output)
        return vanilla_output, bt_output

def my_model_function():
    return MyModel()

def GetInput():
    batch_size = 64
    max_seqlen = 256
    inputs = torch.rand(batch_size, max_seqlen, 768, dtype=torch.float32)
    return inputs

