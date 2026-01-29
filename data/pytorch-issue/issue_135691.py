# torch.rand(S, B, 1000, dtype=torch.float32)  # S: sequence length, B: batch size
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_size = 500
        self.num_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=2 * self.hidden_size,  # 1000
            num_heads=self.num_heads
        )

    def forward(self, input_x):
        # Returns (output, attention_weights)
        return self.attention(input_x, input_x, input_x)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the original export_shape (249, 1, 1000)
    return torch.randn(249, 1, 1000, dtype=torch.float32)

