# torch.rand(S, B, D, dtype=torch.float32)  # Example: S=5, B=2, D=256
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Recompute dim_feedforward for SwiGLU
        dim_feedforward = int(2 * dim_feedforward / 3)
        dim_feedforward = gate_multiple_of * (
            (dim_feedforward + gate_multiple_of - 1) // gate_multiple_of
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(d_model, dim_feedforward)

    def _ff_block(self, x):
        # SwiGLU implementation with trainable parameters
        activation = self.activation(self.linear1(x))
        gate = self.linear3(x)
        return self.dropout2(self.linear2(self.dropout(activation * gate)))

class MyModel(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # Override to bypass sparsity checks causing isinstance errors
        output = src
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output

def my_model_function():
    layer = _TransformerEncoderLayerSwiGLU(
        d_model=256,
        nhead=8,
        dim_feedforward=2048,  # Original dim_feedforward before SwiGLU adjustment
        gate_multiple_of=128,
        dropout=0.1,
        norm_first=False,
    )
    return MyModel(encoder_layer=layer, num_layers=6)

def GetInput():
    # Input shape (sequence_length, batch_size, d_model)
    return torch.rand(5, 2, 256, dtype=torch.float32)

