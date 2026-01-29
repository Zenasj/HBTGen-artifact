# torch.rand(1, 10, dtype=torch.int64)  # Inferred input shape for input_ids

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(50264, 10)  # Example embedding layer
        self.linear = nn.Linear(10, 10)  # Example linear layer

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    vocab_size = 50264
    length = 10
    input_ids = torch.randint(vocab_size, (1, length), dtype=torch.int64).to('cuda' if torch.cuda.is_available() else 'cpu')
    return input_ids

