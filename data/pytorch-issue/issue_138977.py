# torch.rand(B, 512, dtype=torch.long)  # Input shape: batch_size x sequence_length
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mimicking Codet5-base encoder structure based on common Seq2SeqLM patterns
        self.embedding = nn.Embedding(30522, 768)  # Vocabulary size and hidden dimension
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=768, nhead=12) for _ in range(12)
        ])
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.encoder_layers:
            x = layer(x)
        return x  # Return last_hidden_state as in HuggingFace's BaseModelOutput

def my_model_function():
    model = MyModel()
    # Load weights with CPU mapping (uncomment and provide path when using actual model)
    # model.load_state_dict(torch.load('/path/to/pytorch_model.bin', map_location='cpu'))
    return model.to('cpu')  # Explicitly ensure model is on CPU

def GetInput():
    # Generate random input_ids tensor matching expected input dimensions
    return torch.randint(0, 30522, (1, 512), dtype=torch.long)

