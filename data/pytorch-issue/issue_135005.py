import torch.nn as nn

import torch
import torch_tensorrt
from typing import Optional, Sequence,Dict,List
from torch.nn import functional as F
from tzrec.modules.mlp import MLP
from torch import nn

@torch.fx.wrap
def _get_dict(grouped_features_keys: List[str], args:List[torch.Tensor])->Dict[str, torch.Tensor]:
    if len(grouped_features_keys) != len(args):
            raise ValueError(
                "The number of grouped_features_keys must match "
                "the number of arguments."
            )
    grouped_features = {
        key: value for key, value in zip(grouped_features_keys, args)
    }
    return grouped_features

@torch.fx.wrap
def _arange(end: int, device: torch.device) -> torch.Tensor:
    return torch.arange(end, device=device)

class MatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = ["query","sequence","sequence_length"]
        attn_mlp= {'hidden_units': [256, 64], 'dropout_ratio': [], 'activation': 'nn.ReLU', 'use_bn': False}
        self.mlp = MLP(in_features=41 * 4, **attn_mlp)
        self.linear = nn.Linear(self.mlp.hidden_units[-1], 1)

    def forward(self, *args1: List[torch.Tensor]):
        """Forward the module."""
        # use predict to avoid trace error in self._output_to_prediction(y)
        return self.predict(args1)
    
    def predict(self, args: List[torch.Tensor]):
        grouped_features= _get_dict(self.keys, args)
        query = grouped_features["query"]
        sequence = grouped_features["sequence"]
        sequence_length = grouped_features["sequence_length"]
        max_seq_length = sequence.size(1)
        sequence_mask = _arange(
            max_seq_length, device=sequence_length.device
        ).unsqueeze(0) < sequence_length.unsqueeze(1)

       
        queries = query.unsqueeze(1).expand(-1, max_seq_length, -1)

        attn_input = torch.cat(
            [queries, sequence, queries - sequence, queries * sequence], dim=-1
        )
        
        return attn_input
       

model = MatMul().eval().cuda()
a=torch.randn(1, 41).cuda()
b=torch.randn(1, 50,41).cuda()
c=torch.randn(1).cuda()
torch._dynamo.mark_dynamic(a, 0,min=1,max=8196)
torch._dynamo.mark_dynamic(b, 0,min=1,max=8196)
# torch._dynamo.mark_dynamic(b, 1, min=1, max=50)
torch._dynamo.mark_dynamic(c, 0,min=1,max=8196)
inputs = [a, b,c]
print(model(*inputs)[0][0][0])
# seq_len = torch.export.Dim("seq_len", min=1, max=10)
# dynamic_shapes=({2: seq_len}, {2: seq_len})
# Export the model first with custom dynamic shape constraints
from torchrec.fx import symbolic_trace
model = symbolic_trace(model)
print(model.code)
exp_program = torch.export.export(model, (*inputs,))