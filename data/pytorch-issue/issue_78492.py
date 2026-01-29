# torch.rand(B, 2, 80, dtype=torch.long)  # Batch, num_choices, sequence_length
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simulate Roberta-based multiple-choice model structure
        self.embedding = nn.Embedding(50265, 1024)  # Vocabulary size from config
        self.encoder = nn.Sequential(
            nn.Linear(1024, 1024),  # Simplified transformer layer placeholder
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(1024, 1)  # Output per choice
        
    def forward(self, input_ids):
        batch_size, num_choices, seq_len = input_ids.shape
        # Flatten batch and choices for processing
        x = input_ids.view(batch_size * num_choices, seq_len)
        x = self.embedding(x)
        x = x.mean(dim=1)  # Simplified pooling
        x = self.encoder(x)
        logits = self.classifier(x)
        # Reshape to (batch_size, num_choices)
        return logits.view(batch_size, num_choices)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input tensor matching expected shape
    batch_size = 2  # Small batch for testing
    return torch.randint(0, 50265, (batch_size, 2, 80), dtype=torch.long)

