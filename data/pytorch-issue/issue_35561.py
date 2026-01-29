# torch.randint(0, 30522, (1, 128), dtype=torch.int64)
import torch
from transformers import DistilBertModel

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids):
        return self.distilbert(input_ids)[0]  # Return last_hidden_state for consistency

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input_ids with typical BERT vocab size (30522) and sequence length 128
    return torch.randint(0, 30522, (1, 128), dtype=torch.int64)

