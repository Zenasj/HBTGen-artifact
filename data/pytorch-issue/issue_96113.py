import torch
import torch.nn as nn

ln = nn.LayerNorm((768,), elementwise_affine=True).to("mps")
ln(torch.randn(1, 77, 768).to("mps", dtype=torch.float16))