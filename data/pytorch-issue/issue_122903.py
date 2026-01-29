# torch.randint(0, 30000, (BATCH_SIZE, SEQUENCE_LENGTH), dtype=torch.long)  # Example input shape (B=2, S=10)

import torch
import torch.nn as nn

class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 768)
    
    def forward(self, x):
        return self.linear(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30000, 768)  # Vocabulary size and hidden dimension
        self.layers = nn.ModuleList([MyLayer() for _ in range(12)])  # 12 layers for transformer block
        self.output = nn.Linear(768, 30000)  # Output layer to vocabulary size
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

def my_model_function():
    model = MyModel()
    model = model.bfloat16()  # Matches original model's torch_dtype=bfloat16
    return model

def GetInput():
    # Returns input tensor matching shape (batch, sequence_length) for Llama-like model
    return torch.randint(0, 30000, (2, 10), dtype=torch.long)  # Example batch_size=2, seq_len=10

