import torch.nn as nn

import torch
import torch.nn.functional as F
from einops import rearrange


def fn(q, k, v, b, w):

    # apply linear layer to bias
    b = rearrange(b @ w, "B Q K H -> B H Q K")

    # if you comment out this line, the error goes away
    b = b.contiguous()

    # attention
    return F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=b,
    )


if __name__ == "__main__":

    # setup
    DEVICE = torch.device("cuda:0")
    DTYPE = torch.float16
    torch.manual_seed(999)
    B = 3
    H = 8
    Q = 99
    K = 80
    D = 32
    C_bias = 128

    # inputs
    query = torch.randn((B, H, Q, D), device=DEVICE, dtype=DTYPE)
    key = torch.randn((B, H, K, D), device=DEVICE, dtype=DTYPE)
    value = torch.randn((B, H, K, D), device=DEVICE, dtype=DTYPE)
    bias = torch.randn((B, Q, K, C_bias), device=DEVICE, dtype=DTYPE)
    weights = torch.randn((C_bias, H), device=DEVICE, dtype=DTYPE)

    # eager version is okay
    fn(query, key, value, bias, weights)

    # compiled version causes error
    static = torch.compile(fn)
    static(query, key, value, bias, weights)

from transformers import LlamaForCausalLM
import torch
import time
import gc
llm_name = "meta-llama/Llama-2-7b-hf"
llm = LlamaForCausalLM.from_pretrained(llm_name).cuda()

llm.forward = torch.compile(llm.forward)

outputs = llm(torch.ones(1, 450, dtype = torch.long).cuda())
print("------")
outputs = llm(torch.ones(1, 1, dtype = torch.long).cuda())