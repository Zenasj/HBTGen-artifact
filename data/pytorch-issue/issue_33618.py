# torch.rand(1, 10, 512, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(512, 512, batch_first=True)
        self.linear = nn.Linear(512, 32)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel()
    lstm_weights = [p for p in dict(model.lstm.named_parameters()).keys() if "weight" in p]
    for p in lstm_weights:
        prune.l1_unstructured(model.lstm, p, 0.5)
        prune.remove(model.lstm, p)
    model.lstm._apply(lambda x: x)  # Apply the fix to unblock the training
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 10, 512, device="cuda:0", dtype=torch.float32)

