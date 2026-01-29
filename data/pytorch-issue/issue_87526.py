# torch.rand(1, 4, dtype=torch.float32)
import torch
from transformers.tokenization_utils_base import BatchEncoding

class MyModel(torch.nn.Module):
    def forward(self, x):
        encoding = BatchEncoding({'key': x})
        return encoding['key']

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((1, 4), dtype=torch.float32)

