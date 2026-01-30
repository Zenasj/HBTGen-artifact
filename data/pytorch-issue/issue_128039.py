import torch.nn as nn

# take mlp for example
for layer in model.model.layers:
    layer.mlp = Triton_myMLP(layer.mlp)

model = AutoModelForCausalLM.from_pretrained("luodian/llama-7b-hf", torch_dtype=torch.float16)
model = model.to(device).eval()
# replace mlp by my triton operator
# compiled model returns correct output without this 
# ['<s> My favourite condiment is ketchup. I love it on everything. I love it on my eggs, on my burg']
for layer in model.model.layers:
    layer.mlp = Triton_myMLP(layer.mlp)

tokenizer = AutoTokenizer.from_pretrained("luodian/llama-7b-hf")
prompt = "My favourite condiment is"

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
model.generation_config.max_new_tokens = 20
out = model.generate(input_ids, cache_implementation="static") # warmup ?
print(tokenizer.batch_decode(out.long()))
out = model.generate(input_ids, cache_implementation="static")
print(tokenizer.batch_decode(out.long())) # the result is like ['<s> My favourite condiment is plus plus plus plus plus plus plus plus plus plus plus plus plus plus plus plus plus plus plus plus']

for i in range(2):
    text = generate(prompt, model, tokenizer, max_length=20)

print(text) # this gives the correct results

['<s> My favourite condiment is plus plus plus plus plus plus plus plus plus plus plus plus plus plus plus plus plus plus plus plus']

import triton 
from triton import language as tl
import torch

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    # configs=get_cuda_autotune_config(),
    configs= [
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=3,
                      num_warps=16),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=3,
                      num_warps=16),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c

from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import torch
from typing import Optional
from kernels.basic_gemm import matmul # change to your path
from torch import nn
device = "cuda"

class Triton_myMLP(nn.Module):
    def __init__(self, llama_mlp_layer):
        super().__init__()

        W1 = llama_mlp_layer.gate_proj.weight.cuda()
        self.W1T = torch.empty_like(W1.t()).copy_(W1.t()).contiguous().cuda()
        W2 = llama_mlp_layer.up_proj.weight.cuda()
        self.W2T = torch.empty_like(W2.t()).copy_(W2.t()).contiguous().cuda()
        W3 = llama_mlp_layer.down_proj.weight.cuda()
        self.W3T = torch.empty_like(W3.t()).copy_(W3.t()).contiguous().cuda()

    def forward(self, x):
        
        x = x.view(-1, 4096)

        gate = torch.nn.functional.silu(x @ self.W1T)
        up = matmul(x, self.W2T)
        c = gate*up
        output = c @ self.W3T

        return output[None, :, :]
            
model = AutoModelForCausalLM.from_pretrained("luodian/llama-7b-hf", torch_dtype=torch.float16)
model = model.to(device).eval()
# replace mlp
for layer in model.model.layers:
    layer.mlp = Triton_myMLP(layer.mlp)


tokenizer = AutoTokenizer.from_pretrained("luodian/llama-7b-hf")
prompt = "My favourite condiment is"


input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

model.generation_config.max_new_tokens = 20
out = model.generate(input_ids) # warmup ?
print(tokenizer.batch_decode(out.long())) # output is correct
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

out = model.generate(input_ids, cache_implementation="static")
print(tokenizer.batch_decode(out.long()))