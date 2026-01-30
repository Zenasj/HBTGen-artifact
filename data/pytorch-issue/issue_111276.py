import torch.nn as nn

from torch.nn import TransformerEncoderLayer, Linear
import torch
from torch import nn

class DemoModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dense = Linear(512, 10)
    
    def forward(self, x):
        src_query_mask = torch.rand(1, 41)
        src_query_emb_VT = torch.rand(1, 41, 512)
        out1 = self.transformer_encoder(src_query_emb_VT , src_key_padding_mask = src_query_mask)
        res = self.dense(out1)
        return res

src = torch.rand(2, 32, 512)
print(src.shape) # torch.Size([2, 32, 512])
demo_model = DemoModel()
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    out = demo_model(src)
    print("Out shape", out.shape)

from torch.nn import Linear
import torch
from torch import nn

class DemoModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dense = Linear(512, 10)
    
    def forward(self, x):
        src_query_mask = torch.rand(1, 41)
        src_query_emb_VT = torch.rand(1, 41, 512)
        out1 = self.transformer_encoder(src_query_emb_VT , src_key_padding_mask = src_query_mask)
        res = self.dense(out1)
        return res

src = torch.rand((2, 32, 512), device="cuda")
print(src.shape) # torch.Size([2, 32, 512])
demo_model = DemoModel()
demo_model.eval()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    print("src device: {}".format(src.get_device())) # 0, gpu
    out = demo_model(src)
    print("Out shape", out.shape)
    print("out device: ", out.get_device()) # -1, cpu