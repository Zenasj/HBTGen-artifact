import torch.nn as nn

import torch

q = torch.ones([1, 1, 16384, 512], dtype=torch.float16, device="cuda")
k, v = q.clone(), q.clone()

result = torch.nn.functional.scaled_dot_product_attention(q, k, v)