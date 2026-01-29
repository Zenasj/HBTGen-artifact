# torch.rand(1, 16, dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified embedding and output layer for GPT-J-like model
        self.embedding = nn.Embedding(10000, 128)  # Example vocab size and embedding dim
        self.linear = nn.Linear(128, 10000)  # Output layer for token probabilities

    def forward(self, input_ids):
        # Forward pass: embedding → linear projection
        embeddings = self.embedding(input_ids)
        return self.linear(embeddings)

    def generate(self, input_ids, max_length=200, do_sample=True, temperature=0.9):
        # Simple greedy/temperature sampling for demonstration
        outputs = input_ids
        for _ in range(max_length - input_ids.shape[1]):
            next_token_logits = self(outputs[:, -1:])  # Predict next token
            if do_sample:
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
            outputs = torch.cat([outputs, next_token], dim=1)
        return outputs

def my_model_function():
    return MyModel()

def GetInput():
    # Returns input_ids tensor matching expected shape
    return torch.randint(0, 10000, (1, 16), dtype=torch.long)

