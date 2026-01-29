# torch.randint(0, 100, (B, S), dtype=torch.long)  # B=batch_size, S=sequence_length (e.g., (1, 64))
import torch
from transformers import AutoModel
import torch.quantization
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        # Apply dynamic quantization to Linear layers
        torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    
    def forward(self, input_ids):
        # BERT expects input_ids as primary input (other args like attention_mask can be added if needed)
        return self.model(input_ids)[0]  # Return last_hidden_state for simplicity

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input_ids tensor matching BERT's expected input (int64/long type)
    return torch.randint(0, 100, (1, 64), dtype=torch.long)

