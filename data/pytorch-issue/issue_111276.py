# torch.rand(B, 32, 512, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Original model with float mask (causes error on CPU with autocast)
        class DemoModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
                self.dense = nn.Linear(512, 10)
            
            def forward(self, x):
                src_query_mask = torch.rand(1, 41)  # Float mask causing dtype mismatch
                src_query_emb_VT = torch.rand(1, 41, 512)
                out1 = self.transformer_encoder(src_query_emb_VT, src_key_padding_mask=src_query_mask)
                return self.dense(out1)
        
        # Fixed model with bool mask (resolves CPU autocast issue)
        class FixedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
                self.dense = nn.Linear(512, 10)
            
            def forward(self, x):
                src_query_mask = torch.rand(1, 41).bool()  # Convert mask to bool
                src_query_emb_VT = torch.rand(1, 41, 512)
                out1 = self.transformer_encoder(src_query_emb_VT, src_key_padding_mask=src_query_mask)
                return self.dense(out1)
        
        self.original = DemoModel()  # Model with faulty mask
        self.fixed = FixedModel()    # Model with corrected mask
    
    def forward(self, x):
        # Compare outputs of original (faulty) and fixed models
        # Returns 1.0 if outputs match within tolerance, else 0.0
        try:
            out_original = self.original(x)
        except RuntimeError:
            # Handle case where original model fails (e.g., CPU autocast error)
            return torch.tensor(0.0)
        
        out_fixed = self.fixed(x)
        return torch.tensor(1.0) if torch.allclose(out_original, out_fixed, atol=1e-4) else torch.tensor(0.0)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the example's input shape (batch=2, seq_len=32, d_model=512)
    return torch.rand(2, 32, 512)

