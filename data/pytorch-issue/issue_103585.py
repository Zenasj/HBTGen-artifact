# torch.rand(B, S, dtype=torch.long)  # Input shape (batch, sequence_length) with token IDs
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified model structure mimicking GPT-J's problematic component
        self.embedding = nn.Embedding(100, 768)  # Mock token embeddings
        self.linear = nn.Linear(768, 768)
        
        # Inject a problematic torch.bool parameter to reproduce the error
        self.bias_bool = nn.Parameter(torch.zeros(768, dtype=torch.bool), requires_grad=False)
        
        # Add a stub for the causal LM head (matches original model structure)
        self.lm_head = nn.Linear(768, 100, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        # Artificially incorporate the problematic bool parameter into computation
        x = x + self.bias_bool.unsqueeze(0).unsqueeze(1)  # Broadcast to batch/seq dims
        return self.lm_head(x)

def my_model_function():
    # Returns the model with the problematic bool parameter
    return MyModel()

def GetInput():
    # Generate random token IDs (batch=2, sequence_length=10)
    return torch.randint(0, 100, (2, 10), dtype=torch.long)

