# torch.rand(1, 10, 4), torch.rand(1, 20, 4)  # src and tgt input shapes
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=4,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            batch_first=True,
            dropout=0.0,
        )

    def forward(self, inputs):
        src, tgt = inputs
        # Compute full output with full tgt sequence
        tgt_len = tgt.size(1)
        mask_full = self.transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        out_full = self.transformer(src, tgt, tgt_mask=mask_full)
        
        # Compute limited output with first 10 tokens of tgt
        tgt_limited = tgt[:, :10, :]
        mask_limited = self.transformer.generate_square_subsequent_mask(10).to(tgt_limited.device)
        out_limited = self.transformer(src, tgt_limited, tgt_mask=mask_limited)
        
        # Return mean absolute difference between first 10 positions
        return torch.mean(torch.abs(out_full[:, :10] - out_limited))

def my_model_function():
    model = MyModel()
    model.eval()
    return model

def GetInput():
    src = torch.rand(1, 10, 4)
    tgt = torch.rand(1, 20, 4)
    return (src, tgt)

