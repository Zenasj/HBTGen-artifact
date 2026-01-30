import torch.nn as nn

import torch
import copy

num_heads = 16
head_dim = 128
torch.set_printoptions(threshold=1000000, sci_mode=True)

def _attn_sdpa(query, key, value, attention_mask=None, contiguify=False, enable_mem_efficient=False):
    query_shape = query.shape
    batch_size = query_shape[0]
    kv_seq_len = key.shape[-2]

    query_length = query_shape[1]

    # NOTE: Maybe there is better than this?
    query = query.view(batch_size, query_length, num_heads, head_dim).transpose(1, 2)

    # Without these unsqueeze, SDPA complains as the query and key/value have a different number of dimensions.
    key = key.unsqueeze(1)
    value = value.unsqueeze(1)

    # Although these expand are not numerically useful, PyTorch 2.1 can not dispatch to mem-efficient attention
    # and flash attention (No available kernel.  Aborting execution.) from the shapes
    # query = [batch_size, num_heads, query_length, head_dim]
    # key = [batch_size, 1, kv_length, head_dim]
    # value = [batch_size, 1, kv_length, head_dim]
    # which is unfortunate. Hopefully can be improved in the future. These expand should not be too expansive as they do not do memory copy.
    key = key.expand(-1, num_heads, -1, -1)
    value = value.expand(-1, num_heads, -1, -1)


    if contiguify:
        key = key.contiguous()
        value = value.contiguous()

    print("query contiguous", query.is_contiguous())
    print("key contiguous", key.is_contiguous())
    print("value contiguous", value.is_contiguous())

    if enable_mem_efficient:
        enable_math = False
    else:
        enable_math = True

    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=enable_math, enable_mem_efficient=enable_mem_efficient):
        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

    return sdpa_result

device = "cpu"

query_sdpa = torch.load("query_sdpa.pt").to(device)
key_sdpa = torch.load("key_sdpa.pt").to(device)
value_sdpa = torch.load("value_sdpa.pt").to(device)
attention_mask_sdpa = torch.load("attention_mask_sdpa.pt").to(device)

print("query_sdpa", query_sdpa.shape)
print("key_sdpa", key_sdpa.shape)
print("value_sdpa", value_sdpa.shape)
print("attention_mask_sdpa", attention_mask_sdpa.shape)
print("attention_mask_sdpa", attention_mask_sdpa)

print("---- non_contig_cpu_math")
res_non_contig_cpu = _attn_sdpa(query_sdpa, key_sdpa, value_sdpa, attention_mask_sdpa, contiguify=False)
print("---- contig_cpu_math")
res_contig_cpu = _attn_sdpa(query_sdpa, key_sdpa, value_sdpa, attention_mask_sdpa, contiguify=True)

device = "cuda"

query_sdpa = torch.load("query_sdpa.pt").to(device)
key_sdpa = torch.load("key_sdpa.pt").to(device)
value_sdpa = torch.load("value_sdpa.pt").to(device)
attention_mask_sdpa = torch.load("attention_mask_sdpa.pt").to(device)

print("---- non_contig_cuda_math")
res_non_contig_cuda_math = _attn_sdpa(query_sdpa, key_sdpa, value_sdpa, attention_mask_sdpa, contiguify=False)
print("---- contig_cuda_math")
res_contig_cuda_math = _attn_sdpa(query_sdpa, key_sdpa, value_sdpa, attention_mask_sdpa, contiguify=True)

print("---- non_contig_cuda_memeff")
res_non_contig_cuda_memeff = _attn_sdpa(query_sdpa, key_sdpa, value_sdpa, attention_mask_sdpa, contiguify=False, enable_mem_efficient=True)
print("---- contig_cuda_memeff")
res_contig_cuda_memeff = _attn_sdpa(query_sdpa, key_sdpa, value_sdpa, attention_mask_sdpa, contiguify=True, enable_mem_efficient=True)

def print_diff(text, tensor1, tensor2):
    print(f"{text}: mean abs-diff", (tensor1 - tensor2).abs().mean())
    print(f"{text}: mean rel-diff", ((tensor1 - tensor2).abs() / (tensor1.abs() + 1e-12)).mean())

print("\n")
print_diff("cpu non-contig/contig", res_non_contig_cpu, res_contig_cpu)
print_diff("cuda non-contig/contig math", res_non_contig_cuda_math, res_contig_cuda_math)
print_diff("cuda non-contig/contig memeff", res_non_contig_cuda_memeff, res_contig_cuda_memeff)

print("\nAllclose CPU non-contig/contig:", torch.allclose(res_non_contig_cpu, res_contig_cpu))
print("Allclose CUDA math non-contig/contig:", torch.allclose(res_non_contig_cuda_math, res_contig_cuda_math))
print("Allclose CUDA memeff non-contig/contig:", torch.allclose(res_non_contig_cuda_memeff, res_contig_cuda_memeff))

from transformers import GPTBigCodeForCausalLM, AutoTokenizer
import torch

with torch.device("cuda"):
    model = GPTBigCodeForCausalLM.from_pretrained("TabbyML/SantaCoder-1B")
tokenizer = AutoTokenizer.from_pretrained("TabbyML/SantaCoder-1B")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

model = model.eval()
inp = tokenizer(["def quick_sort(", "a =", "print('I"], padding=True, return_tensors="pt").to("cuda")

# This one works.
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    res = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=60, max_new_tokens=60)

for dec in tokenizer.batch_decode(res):
    print("---- math")
    print(dec)

# This one errors out.
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
    res = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=60, max_new_tokens=60)

for dec in tokenizer.batch_decode(res):
    print("---- mem-efficient")
    print(dec)