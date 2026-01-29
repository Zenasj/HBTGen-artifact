# torch.rand(B, 128, 768, dtype=torch.float32)  # B: batch, 128: sequence length, 768: hidden size

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooler = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, 2)  # 2 classes for MRPC task

    def forward(self, x):
        # Global average pooling over sequence length
        pooled = x.mean(dim=1)
        pooled = self.pooler(pooled).tanh()  # Matches BertPooler's activation
        return self.classifier(pooled)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Matches num_processes=2 in config.yaml
    return torch.rand(B, 128, 768, dtype=torch.float32)  # Random input tensor matching BERT's hidden size

