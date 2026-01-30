import torch.nn as nn

from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

import os
torch._inductor.config.fallback_random = True


# torch._dynamo.config.base_dir = os.environ["TORCHINDUCTOR_CACHE_DIR"]

# disable cuda sdp, before we implement it with xpu backend
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)



from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, L_L_self_modules_self_attn_modules_out_proj_parameters_weight_ : torch.Tensor, L_L_self_modules_self_attn_modules_out_proj_parameters_bias_ : torch.Tensor, key_states, value_states, query_states_1):
        l_l_self_modules_self_attn_modules_out_proj_parameters_weight_ = L_L_self_modules_self_attn_modules_out_proj_parameters_weight_
        l_l_self_modules_self_attn_modules_out_proj_parameters_bias_ = L_L_self_modules_self_attn_modules_out_proj_parameters_bias_
        attn_output = torch._C._nn.scaled_dot_product_attention(query_states_1, key_states, value_states, attn_mask = None, dropout_p = 0.0, is_causal = True);  query_states_1 = key_states = value_states = None
        attn_output_1 = attn_output.transpose(1, 2);  attn_output = None
        attn_output_2 = attn_output_1.reshape(1, 1024, 1024);  attn_output_1 = None
        attn_output_3 = torch._C._nn.linear(attn_output_2, l_l_self_modules_self_attn_modules_out_proj_parameters_weight_, l_l_self_modules_self_attn_modules_out_proj_parameters_bias_);  attn_output_2 = l_l_self_modules_self_attn_modules_out_proj_parameters_weight_ = l_l_self_modules_self_attn_modules_out_proj_parameters_bias_ = None
        return (attn_output_3,)


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('18f6e31c9f38831ae36b922d77c4c0546a0b5bb5', 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1024, 1024), requires_grad=True, is_leaf=True)  # L_L_self_modules_self_attn_modules_out_proj_parameters_weight_
    buf1 = reader.storage('1ceaf73df40e531df3bfb26b4fb7cd95fb7bff1d', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1024,), requires_grad=True, is_leaf=True)  # L_L_self_modules_self_attn_modules_out_proj_parameters_bias_
    buf2 = reader.storage('9f89f5ef3ecccc9aa2d2d9ec5d6882b315806934', 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf2, (1, 16, 1024, 64), dtype=torch.float16, requires_grad=True)  # key_states
    buf3 = reader.storage('74883a0ed186fbeb2dbe8b5c549db706a634fdb0', 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf3, (1, 16, 1024, 64), dtype=torch.float16, requires_grad=True)  # value_states
    buf4 = reader.storage('a22f237f6ff537f598c9f0b2728e44c66b52a875', 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf4, (1, 16, 1024, 64), dtype=torch.float16, requires_grad=True)  # query_states_1
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='run',
        save_dir='/home/yunfei/code/piece/checkpoints', autocast=True, backend='inductor')

import torch

from torch.nn import *
class Repro(torch.nn.Module):
    def forward(self, a, b, c):
        attn_output = torch._C._nn.scaled_dot_product_attention(a, b, c, attn_mask = None, dropout_p = 0.0, is_causal = True);
        return attn_output


mod = Repro()

# needs true for repro
requires_grad=True

buf0 = torch.zeros((1, 16, 1024, 64), dtype=torch.float16, requires_grad=requires_grad, device='cuda')
buf1 = torch.zeros((1, 16, 1024, 64), dtype=torch.float16, requires_grad=requires_grad, device='cuda')
buf2 = torch.zeros((1, 16, 1024, 64), dtype=torch.float16, requires_grad=requires_grad, device='cuda')

with torch.amp.autocast("cuda", enabled=True):
    a = mod(buf0, buf1, buf2)
    b = torch.compile(mod)(buf0, buf1, buf2)
    assert a.dtype == b.dtype