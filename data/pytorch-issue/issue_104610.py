# torch.randint(0, 100, (1, 20), dtype=torch.long)  # Example input shape (batch_size, sequence_length)
import torch
from transformers import GPT2Model

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained('gpt2')

    def forward(self, input_ids):
        return self.gpt(input_ids)[0]  # Return last_hidden_state for simplicity

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input tensor compatible with MyModel
    return torch.randint(0, 100, (1, 20), dtype=torch.long)

