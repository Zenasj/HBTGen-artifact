# torch.rand(B, 1024, dtype=torch.long)
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def forward(self, input_ids):
        return self.model(input_ids=input_ids).logits  # Return logits for compatibility

def my_model_function():
    return MyModel()

def GetInput():
    # Batch size 16 (standard) and sequence length 1024 as per logs
    return torch.randint(50257, (16, 1024), dtype=torch.long)

