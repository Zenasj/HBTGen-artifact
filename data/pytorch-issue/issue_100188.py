import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
class MyCustomerLayer(TransformerEncoderLayer):
    pass

encoder = TransformerEncoder(MyCustomerLayer(d_model=256, nhead=8), num_layers=6)
torch.jit.script(encoder)

class _TransformerEncoderLayerSwiGLU(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        norm_first: bool = False,
        gate_multiple_of: int = 128,
        **_kwargs,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.silu,
            norm_first=norm_first,
        )
        # Reference:
        #     https://github.com/facebookresearch/llama/blob/main/llama/model.py
        dim_feedforward = int(2 * dim_feedforward / 3)
        dim_feedforward = gate_multiple_of * (
            (dim_feedforward + gate_multiple_of - 1) // gate_multiple_of
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(d_model, dim_feedforward)

    def _ff_block(self, x):
        x = self.linear2(
            self.dropout(self.activation(self.linear1(x)) * self.linear3(x))
        )
        return self.dropout2(x)

class _SwiGLU(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()
        self.linear3 = nn.Linear(d_model, dim_feedforward)

    def forward(self, x):
        return F.silu(x) * self.linear3(x)


def get_swiglu_encoder_layer(
    d_model: int,
    nhead: int,
    dim_feedforward: int,
    gate_multiple_of: int = 128,
    **kwargs,
):
    # Reference:
    #     https://github.com/facebookresearch/llama/blob/main/llama/model.py
    dim_feedforward = int(2 * dim_feedforward / 3)
    dim_feedforward = gate_multiple_of * (
        (dim_feedforward + gate_multiple_of - 1) // gate_multiple_of
    )
    activation = _SwiGLU(d_model, dim_feedforward)
    return nn.TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, activation=activation, **kwargs
    )