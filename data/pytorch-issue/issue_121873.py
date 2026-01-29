# torch.randint(0, 30000, (B, 256), dtype=torch.long)  # Input shape: (batch_size, sequence_length)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mimicking Mixtral MoE structure with minimal components
        self.embed_tokens = nn.Embedding(30000, 2560)  # Example vocab size and embedding dim
        self.transformer_layer = nn.Sequential(
            nn.Linear(2560, 2560),
            nn.ReLU(),
            nn.Linear(2560, 2560)
        )
        self.lm_head = nn.Linear(2560, 30000, bias=False)

    def forward(self, input_ids):
        embeddings = self.embed_tokens(input_ids)
        hidden_states = self.transformer_layer(embeddings)
        return self.lm_head(hidden_states)

def my_model_function():
    model = MyModel()
    # Initialize weights (simplified)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    return model

def GetInput():
    # Generate random input matching the model's expected shape
    batch_size = 1  # Matches user's per_device_train_batch_size=1
    return torch.randint(0, 30000, (batch_size, 256), dtype=torch.long)

