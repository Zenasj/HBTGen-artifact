# torch.rand(300, 1, 25, 16, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_size = 16
        self.filter_num = 512
        self.padding_length = 25
        self.convolutions = nn.ModuleList([
            nn.Conv2d(1, self.filter_num // 8, kernel_size=(K, self.embedding_size), stride=1)
            for K in range(1, 9)
        ])

    def forward(self, x):
        conv_outputs = []
        for conv in self.convolutions:
            conv_out = conv(x)
            # Squeeze spatial dimension (since kernel covers entire embedding_size=16)
            squeezed = conv_out.squeeze(3)
            tanhed = torch.tanh(squeezed)
            conv_outputs.append(tanhed)
        
        pooled = []
        for feat in conv_outputs:
            # Pool over the remaining spatial dimension
            pooled_feat = F.max_pool1d(feat, feat.size(2)).squeeze(2)
            pooled.append(pooled_feat)
        
        return torch.cat(pooled, dim=1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(300, 1, 25, 16, dtype=torch.float32)

