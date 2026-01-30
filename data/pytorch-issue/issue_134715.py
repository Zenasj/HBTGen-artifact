import torch.nn as nn

import torch
import torch.nn.functional as F
from torch import nn

torch.use_deterministic_algorithms(True)

dim = 1536
hidden_dim = 4096
seqlen = 333  # Define seqlen as a variable

w1 = nn.Linear(dim, hidden_dim * 2, bias=False)
w2 = nn.Linear(hidden_dim, dim, bias=False)
w1, w2 = w1.cuda(), w2.cuda()

def swiglu_forward(w1, w2, x):
    x, gate = w1(x).chunk(2, dim=-1)
    return w2(F.silu(x) * gate)

x = torch.randn(1, seqlen, dim).cuda()

with torch.autocast("cuda", dtype=torch.bfloat16):
    x = x.to(torch.bfloat16)
    output_full = swiglu_forward(w1, w2, x)
    x_padded = F.pad(x, (0, 0, 0, (4 - seqlen % 4) % 4))
    chunks = x_padded.chunk(4, dim=1)

    chunk_outputs = [swiglu_forward(w1, w2, chunk) for chunk in chunks]
    output_chunked = torch.cat(chunk_outputs, dim=1)
    output_chunked = output_chunked[:, :seqlen, :]

    torch.testing.assert_close(output_full, output_chunked, rtol=1e-3, atol=1e-3)
    print("Test passed successfully!")