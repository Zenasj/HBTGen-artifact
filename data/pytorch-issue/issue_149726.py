# Input: query (2,3,4,8), key (2,3,4,8), value (2,3,4,8), mask (2,1,4,4) on cuda:0
import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

def manual_attention(query, key, value, mask, dropout=0.0):
    scores = torch.matmul(query, key.transpose(3, 2))
    scores += mask
    attn_weights = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=True)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output

class MyModel(nn.Module):
    def forward(self, inputs, dropout_p=0.0):
        query, key, value, mask = inputs
        torch.manual_seed(0)
        manual_out = manual_attention(query, key, value, mask, dropout_p)
        torch.manual_seed(0)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            sdpa_out = scaled_dot_product_attention(
                query, key, value, attn_mask=mask, is_causal=False, dropout_p=dropout_p, scale=1.0
            )
        return torch.abs(manual_out - sdpa_out).mean()

def my_model_function():
    return MyModel()

def GetInput():
    torch.manual_seed(0)
    query = torch.randn(2, 3, 4, 8, device="cuda:0")
    key = torch.randn(2, 3, 4, 8, device="cuda:0")
    value = torch.randn(2, 3, 4, 8, device="cuda:0")
    mask = torch.where(
        torch.rand(2, 1, 4, 4, device="cuda:0") > 0.5,
        0.0,
        -float("inf")
    )
    return (query, key, value, mask)

