# torch.rand(B, S, C, dtype=torch.float32)  # B=batch, S=sequence length, C=n_features
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, n_features, emb_size):
        super(MyModel, self).__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=emb_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=emb_size, hidden_size=n_features, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded

def my_model_function():
    # Example parameters matching the original model initialization
    n_features = 5  # Example value (replace with actual feature count)
    emb_size = 10   # Example embedding size (replace with actual)
    return MyModel(n_features, emb_size)

def GetInput():
    # Generate random input tensor with example dimensions
    B, S, C = 2, 10, 5  # Batch=2, Sequence length=10, Features=5
    return torch.rand(B, S, C, dtype=torch.float32)

