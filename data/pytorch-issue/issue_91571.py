# torch.rand(B, S, D, dtype=torch.float)  # B=batch, S=sequence length, D=512
import torch
import torch.nn as nn
import torch.quantization

class MyModel(nn.Module):
    class TransformerDecoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dropout=0.1):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        def forward(self, x):
            return self.self_attn(x, x, x)[0]  # Return attention output
        
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.original = MyModel.TransformerDecoderLayer(d_model, nhead)
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        self.original.qconfig = quantization_config
        self.quantized = torch.quantization.prepare(self.original, inplace=False)
    
    def forward(self, x):
        # Compare outputs of original and quantized models
        out_original = self.original(x)
        out_quantized = self.quantized(x)
        return torch.tensor([torch.allclose(out_original, out_quantized)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 10, 512, dtype=torch.float)  # B=2, S=10, D=512

