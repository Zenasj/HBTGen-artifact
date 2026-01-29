# torch.rand(B, S, E, dtype=torch.float32)  # Inputs are (batch, seq_len, embed_dim)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            dim_feedforward=32,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(16)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            norm=encoder_norm,
            enable_nested_tensor=True,
        )
        # Precompute masks as part of the model based on the issue's example
        self.mask = torch.zeros(3, 3, dtype=torch.bool, device='cuda')
        self.padding_mask = torch.tensor(
            [[False, False, False], [False, False, True]],
            dtype=torch.bool,
            device='cuda'
        )
    
    def forward(self, inputs):
        return self.encoder(
            inputs,
            mask=self.mask,
            src_key_padding_mask=self.padding_mask
        )

def my_model_function():
    return MyModel().to('cuda')

def GetInput():
    return torch.randn(2, 3, 16, dtype=torch.float32, device='cuda')

