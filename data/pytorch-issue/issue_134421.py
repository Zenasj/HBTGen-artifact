import torch.nn as nn

import torch
import torch.ao.quantization.fx

class Qdq(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, 
            x, x_scale, x_zp
            ):
        x_q = torch.ops.quantized_decomposed.quantize_per_tensor(     # type: ignore
                x, 
                x_scale,
                x_zp,
                -127,
                127,
                torch.int8)
        x_dq = torch.ops.quantized_decomposed.dequantize_per_tensor(  # type: ignore
                x_q, 
                x_scale,
                x_zp,
                -127,
                127,
                torch.int8)
        return x_dq


p_qdq = torch.export.export(   # type: ignore
        Qdq().eval(), 
        (
            torch.randn([4,4]),
            torch.tensor(1/127, dtype=torch.float),
            torch.tensor(0, dtype=torch.float)
        )
    )