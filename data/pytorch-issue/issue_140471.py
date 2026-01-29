# torch.randint(0, 30000, (B, 1024), dtype=torch.long)  # Input shape: batch_size x sequence_length
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mimicking a causal LM structure with embedding and linear layers
        self.embedding = nn.Embedding(30000, 768)  # Vocabulary size and hidden size
        self.linear = nn.Linear(768, 30000)         # Output layer for logits

    def forward(self, input_ids):
        # Forward pass through embedding and linear layer
        embedded = self.embedding(input_ids)
        return self.linear(embedded)

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the model's expected input
    batch_size = 4   # Matches per_device_train_batch_size from the issue's code
    seq_length = 1024  # model_max_length from CustomTrainingArguments
    return torch.randint(0, 30000, (batch_size, seq_length), dtype=torch.long)

