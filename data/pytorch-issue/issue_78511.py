import torch

batch_size = 32
n = 20
hidden = 768
num_attention_heads = 12
attention_head_size = hidden // num_attention_heads

def transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

def attention(query, key, *, workaround=False):
    key = transpose_for_scores(key)
    query = transpose_for_scores(query)

    if workaround:
        return torch.matmul(query, key.contiguous().transpose(-1, -2))
    else:
        return torch.matmul(query, key.transpose(-1, -2))

A = torch.randn(batch_size, n, hidden)
B = torch.randn(batch_size, n, hidden)

A_mps = A.to("mps")
B_mps = B.to("mps")

print(torch.allclose(attention(A, B), attention(A_mps, B_mps).cpu()))
print(torch.allclose(attention(A, B), attention(A_mps, B_mps, workaround=True).cpu()))

False
True

True
True