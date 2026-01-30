import torch
print(torch.__version__)
input = torch.rand([8, 8, 0, 2, 3, 13], dtype=torch.float32)
res = torch.fbgemm_linear_quantize_weight(
    input
    )