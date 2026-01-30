import torch.nn.functional as F
import torch.nn as nn
import torch


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)


qmodel = TransformerDecoderLayer(512, 8)
print(qmodel.self_attn.batch_first)
quantization_config = torch.quantization.get_default_qconfig("fbgemm")
qmodel.qconfig = quantization_config
qmodel = torch.quantization.prepare(qmodel, inplace=False)
print(qmodel.self_attn.batch_first)