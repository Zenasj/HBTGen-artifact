# torch.rand(B, C, H, W, dtype=...)  # The input shape is (batch_size, sequence_length) with dtype=torch.int64
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedd = nn.EmbeddingBag(500, 12)

    def forward(self, x_user):
        user = self.embedd(x_user)
        return user

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 10
    sequence_length = 5
    return torch.randint(0, 7, size=(batch_size, sequence_length), dtype=torch.int64)

