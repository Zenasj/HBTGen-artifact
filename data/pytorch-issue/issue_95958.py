# torch.rand(2, 3, 16), torch.randint(1, 4, (2,))  # Example input shapes
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, norm=encoder_norm)

    def forward(self, x):
        inputs, input_seq_len = x
        seq_len = inputs.size(1)
        device = inputs.device
        # Create causal mask (src_mask)
        src_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).triu_(diagonal=1)
        # Create padding mask
        padding_mask = (torch.arange(seq_len, device=device)[None, :] >= input_seq_len[:, None])
        return self.encoder(inputs, mask=src_mask, src_key_padding_mask=padding_mask)

def my_model_function():
    return MyModel()

def GetInput():
    B, S, D = 2, 3, 16
    inputs = torch.randn(B, S, D)
    input_seq_len = torch.randint(1, S+1, (B,))
    return (inputs, input_seq_len)

