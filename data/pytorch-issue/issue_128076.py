# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (batch_size, sequence_length)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, vocab_size)
        self.decoder.weight = self.embedding.weight  # Tying weights

    def forward(self, input):
        embedded = self.embedding(input)
        output = self.decoder(embedded)
        return output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    vocab_size = 10000
    embedding_dim = 300
    model = MyModel(vocab_size, embedding_dim)
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 8
    sequence_length = 10
    vocab_size = 10000
    input_tensor = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return input_tensor

