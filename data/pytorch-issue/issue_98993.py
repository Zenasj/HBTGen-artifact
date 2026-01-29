# torch.randint(B, S, dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mock base model mimicking HuggingFace's LlamaForCausalLM
        self.base_model = nn.Sequential(
            nn.Embedding(30522, 1024),  # Llama uses 30522 tokens, hidden_size 1024 (simplified)
            nn.Linear(1024, 1024),
        )
        # Add dummy config to replicate PEFT's base_model.config access
        self.base_model.config = type('Config', (object,), {'hidden_size': 1024, 'num_layers': 32})()
        
        # Mock PEFT LoRA layers (simple linear for demonstration)
        self.lora_layer = nn.Linear(1024, 30522)  # Output to vocab size

    def forward(self, input_ids):
        # Simulate PEFT's forward logic that accesses base_model.config
        hidden_states = self.base_model(input_ids)
        # Example of config access (triggering Dynamo issue)
        _ = self.base_model.config.hidden_size
        return self.lora_layer(hidden_states)

def my_model_function():
    # Returns a simplified PEFT-wrapped model instance
    return MyModel()

def GetInput():
    # Generate random input matching Llama's expected input (batch, seq_length)
    batch_size = 1
    seq_length = 8  # Typical for short inputs
    return torch.randint(0, 30522, (batch_size, seq_length), dtype=torch.long)

