# torch.randint(30522, (1, 512), dtype=torch.long)
import torch
from transformers import BertConfig, BertModel
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        config = BertConfig(num_hidden_layers=1)
        self.bert = BertModel(config)
        self.bert.eval()  # Matches original code's model.eval()

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        return outputs.last_hidden_state, outputs.pooler_output  # Matches ONNX output names

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(30522, (1, 512), dtype=torch.long)

