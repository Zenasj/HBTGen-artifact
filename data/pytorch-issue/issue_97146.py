# torch.randint(0, 30522, (B, S), dtype=torch.long)  # B=batch, S=sequence length
import torch
from torch import nn
from transformers import AutoModel

class MyModel(nn.Module):
    def __init__(self, model_path: str = "distilbert-base-uncased", finetune: bool = False):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(model_path)
        if not finetune:
            self.text_model.eval()
            for param in self.text_model.parameters():
                param.requires_grad_(False)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.text_model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

def my_model_function():
    return MyModel(finetune=False)

def GetInput():
    B, S = 2, 128  # Batch size and sequence length
    return torch.randint(0, 30522, (B, S), dtype=torch.long)

