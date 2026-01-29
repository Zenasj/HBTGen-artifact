import random
import torch
from torch import nn

# torch.randint(0, 100, (1, 512), dtype=torch.int64)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        # Dummy outputs mimicking HuggingFace CausalLM outputs
        logits = torch.randn(batch_size, seq_len, 100)  # vocab_size=100 placeholder
        past_key_values_cache = torch.randn(batch_size, seq_len, 512)  # placeholder tensor
        return logits, past_key_values_cache

def my_model_function():
    return MyModel()

def GetInput():
    # Select a random sequence length from the user's test cases
    seq_len = random.choice([512, 1024, 2048])
    return torch.randint(0, 100, (1, seq_len), dtype=torch.int64)

