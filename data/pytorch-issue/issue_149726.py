import torch
import torch.nn as nn

py
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel


def manual_attention(query, key, value, mask, dropout=0.0):
    scores = torch.matmul(query, key.transpose(3, 2))
    scores += mask
    attn_weights = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=True)

    attn_output = torch.matmul(attn_weights, value)

    return attn_output


def compare(query, key, value, mask, dropout=0.0, backends: list = []):
    torch.manual_seed(0)
    manual_result = manual_attention(query, key, value, mask=mask, dropout=dropout)
    torch.manual_seed(0)
    with sdpa_kernel(backends=backends):
        sdpa_result = scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=False, dropout_p=dropout, scale=1.0
        )

    return torch.abs(manual_result - sdpa_result).mean()


torch.manual_seed(0)
query = torch.randn(2, 3, 4, 8, device="cuda:0")
key = torch.randn(2, 3, 4, 8, device="cuda:0")
value = torch.randn(2, 3, 4, 8, device="cuda:0")
mask = torch.where(torch.rand(2, 1, 4, 4, device="cuda:0") > 0.5, 0.0, -float("inf"))

print(compare(query, key, value, mask=mask, dropout=0.0, backends=[SDPBackend.EFFICIENT_ATTENTION])) # tensor(1.0005e-07, device='cuda:0')
print(compare(query, key, value, mask=mask, dropout=0.5, backends=[SDPBackend.EFFICIENT_ATTENTION])) # tensor(0.9543, device='cuda:0')
print(compare(query, key, value, mask=mask, dropout=0.0, backends=[SDPBackend.MATH])) # tensor(0., device='cuda:0')
print(compare(query, key, value, mask=mask, dropout=0.5, backends=[SDPBackend.MATH])) # tensor(0., device='cuda:0')