# torch.rand(B, 1, 144, 39, dtype=torch.float32)  # x shape. Targets are part of the input tuple (shape (48, 144)).
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, input_dim=16, output_dim=32):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, output_dim, 3, 1, padding="same")
        self.out = torch.nn.Linear(input_dim * output_dim, output_dim)

    def forward(self, inputs):
        x, targets = inputs
        x = self.conv(x)  # (batch_size, channels, seq_len, input_dim)
        b, c, t, f = x.size()
        x = self.out(x.reshape(b, t, c * f))
        # Permute for loss
        logits = x.reshape(x.size(0), x.size(2), x.size(1))
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return loss

def my_model_function():
    return MyModel(input_dim=39, output_dim=111)

def GetInput():
    batch_size = 48
    seq_len = 144
    input_dim = 39
    num_classes = 111
    x = torch.rand(batch_size, 1, seq_len, input_dim, dtype=torch.float32)
    targets = torch.randint(0, num_classes - 1, (batch_size, seq_len), dtype=torch.int64)
    return (x, targets)

