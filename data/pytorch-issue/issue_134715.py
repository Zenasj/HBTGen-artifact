# torch.rand(1, 333, 1536, dtype=torch.bfloat16)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 1536
        hidden_dim = 4096
        self.w1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def swiglu_forward(self, x):
        x, gate = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x) * gate)

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_full = self.swiglu_forward(x)
            # Compute chunked output
            seqlen = x.size(1)
            pad = (4 - seqlen % 4) % 4
            x_padded = F.pad(x, (0, 0, 0, pad))
            chunks = x_padded.chunk(4, dim=1)
            chunk_outputs = [self.swiglu_forward(chunk) for chunk in chunks]
            output_chunked = torch.cat(chunk_outputs, dim=1)
            output_chunked = output_chunked[:, :seqlen, :]
            # Check closeness
            are_close = torch.allclose(
                output_full, output_chunked, rtol=1e-3, atol=1e-3
            )
        return torch.tensor(are_close, dtype=torch.bool, device=x.device)

def my_model_function():
    model = MyModel()
    model.cuda()
    return model

def GetInput():
    return torch.randn(1, 333, 1536, dtype=torch.bfloat16, device="cuda")

