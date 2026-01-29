# torch.rand(B, S, dtype=torch.long)  # B=batch size, S=sequence length (e.g., 10)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholder embedding layer matching Llama-2's typical architecture (vocab size ~30k, 4096 embedding dim)
        # Simplified to smaller dims for minimal repro
        self.embedding = nn.Embedding(num_embeddings=30000, embedding_dim=128)
        self.transformer_layer = nn.Linear(128, 128)  # Simplified transformer-like layer
        self.lm_head = nn.Linear(128, 30000)  # Final prediction layer

    def forward(self, input_ids):
        # Simplified forward pass mimicking causal LM behavior
        embeddings = self.embedding(input_ids)
        hidden_states = self.transformer_layer(embeddings)
        # Average across sequence dimension for minimal reproducibility
        pooled = hidden_states.mean(dim=1)
        return self.lm_head(pooled)

def my_model_function():
    # Matches torch_dtype=torch.bfloat16 from the original script
    model = MyModel().to(torch.bfloat16)
    return model

def GetInput():
    B = 2  # Batch size from original example
    S = 10  # Example sequence length (can vary but needs to be fixed for input generation)
    return torch.randint(0, 30000, (B, S), dtype=torch.long)

