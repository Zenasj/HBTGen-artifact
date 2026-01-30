import torch.nn as nn

Python
import torch

from torch.nn.attention import TensorBias

@torch.compile(fullgraph=True)
def main():
    q = torch.rand(1, 1, 16, 16, dtype=torch.float16, device="cuda")
    k = torch.rand(1, 1, 16, 16, dtype=torch.float16, device="cuda")
    v = torch.rand(1, 1, 16, 16, dtype=torch.float16, device="cuda")

    bias = TensorBias(torch.rand(1, 1, 16, 16, dtype=torch.float16, device="cuda"))

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, bias)
if __name__ == "__main__":
    main()