# torch.rand(B, 512, dtype=torch.int64)
import torch
from transformers import BertForMaskedLM
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def forward(self, input_ids):
        return self.bert(input_ids=input_ids)[0]  # Return logits to match original usage

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size from error case (2,512)
    return torch.randint(0, 30522, (B, 512), dtype=torch.int64)

