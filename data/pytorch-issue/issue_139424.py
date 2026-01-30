import torch.nn as nn

"""Demonstrate torch.compile error on transformers.LlamaForCausalLM scaled_dot_product_attention when providing attention_mask.

To run:

pip install torch@https://download.pytorch.org/whl/nightly/cu124/torch-2.6.0.dev20241030%2Bcu124-cp310-cp310-linux_x86_64.whl pytorch-triton@https://download.pytorch.org/whl/nightly/pytorch_triton-3.1.0%2Bcf34004b8a-cp310-cp310-linux_x86_64.whl transformers==4.46.1
CUDA_VISIBLE_DEVICES=0 python sdpa_compile_error.py
"""

import torch
import transformers
import transformers.models.llama.modeling_llama

BATCH_SIZE = 2
SEQUENCE_LENGTH = 8192


def main() -> None:
    torch.cuda.set_device(device := torch.device("cuda:0"))

    with device:
        config = transformers.LlamaConfig(num_hidden_layers=2)
        llama_model = transformers.LlamaForCausalLM(config)
        llama_model = torch.compile(llama_model)  # <-- works when not compiled

        input_ids = torch.randint(low=0, high=config.vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH))
        attention_mask = torch.ones((BATCH_SIZE, 1), dtype=torch.bool)

    output = llama_model(
        input_ids=input_ids,
        attention_mask=attention_mask,  # <-- works when attention_mask is not provided
        use_cache=False,
    )

    print("done!")


if __name__ == "__main__":
    main()

import torch
import torch._dynamo
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
from torch import device
from torch.nn import *


class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, L_input_ids_ : torch.Tensor, L_self_modules_embed_tokens_parameters_weight_ : torch.nn.parameter.Parameter, L_attention_mask_ : torch.Tensor, L_self_modules_rotary_emb_buffers_inv_freq_ : torch.Tensor, L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_norm_parameters_weight_ : torch.nn.parameter.Parameter):
        l_input_ids_ = L_input_ids_
        l_self_modules_embed_tokens_parameters_weight_ = L_self_modules_embed_tokens_parameters_weight_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_rotary_emb_buffers_inv_freq_ = L_self_modules_rotary_emb_buffers_inv_freq_
        l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        inputs_embeds = torch.nn.functional.embedding(l_input_ids_, l_self_modules_embed_tokens_parameters_weight_, None, None, 2.0, False, False);  l_input_ids_ = l_self_modules_embed_tokens_parameters_weight_ = None
        cache_position = torch.arange(0, 8192, device = device(type='cuda', index=0))
        position_ids = cache_position.unsqueeze(0)
        causal_mask = torch.full((8192, 1), fill_value = -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0))
        causal_mask_1 = torch.triu(causal_mask, diagonal = 1);  causal_mask = None
        arange_1 = torch.arange(1, device = device(type='cuda', index=0))
        reshape = cache_position.reshape(-1, 1);  cache_position = None
        gt = arange_1 > reshape;  arange_1 = reshape = None
        causal_mask_1 *= gt;  causal_mask_2 = causal_mask_1;  causal_mask_1 = gt = None
        getitem = causal_mask_2[(None, None, slice(None, None, None), slice(None, None, None))];  causal_mask_2 = None
        causal_mask_3 = getitem.expand(2, 1, -1, -1);  getitem = None
        causal_mask_4 = causal_mask_3.clone();  causal_mask_3 = None
        getitem_1 = causal_mask_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 1, None))]
        getitem_2 = l_attention_mask_[(slice(None, None, None), None, None, slice(None, None, None))];  l_attention_mask_ = None
        padding_mask = getitem_1 + getitem_2;  getitem_1 = getitem_2 = None
        padding_mask_1 = padding_mask == 0;  padding_mask = None
        getitem_3 = causal_mask_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 1, None))]
        masked_fill = getitem_3.masked_fill(padding_mask_1, -3.4028234663852886e+38);  getitem_3 = padding_mask_1 = None
        causal_mask_4[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 1, None))] = masked_fill;  setitem = causal_mask_4;  masked_fill = setitem = None
        eq_1 = causal_mask_4 == -3.4028234663852886e+38
        all_1 = torch.all(eq_1, dim = -1, keepdim = True);  eq_1 = None
        invert = ~all_1;  all_1 = None
        causal_mask_5 = causal_mask_4.mul(invert);  causal_mask_4 = invert = None
        _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
        getitem_4 = l_self_modules_rotary_emb_buffers_inv_freq_[(None, slice(None, None, None), None)];  l_self_modules_rotary_emb_buffers_inv_freq_ = None
        float_1 = getitem_4.float();  getitem_4 = None
        inv_freq_expanded = float_1.expand(1, -1, 1);  float_1 = None
        getitem_5 = position_ids[(slice(None, None, None), None, slice(None, None, None))];  position_ids = None
        position_ids_expanded = getitem_5.float();  getitem_5 = None
        _enter_autocast = torch.amp.autocast_mode._enter_autocast('cuda', None, False, None)
        float_3 = inv_freq_expanded.float();  inv_freq_expanded = None
        float_4 = position_ids_expanded.float();  position_ids_expanded = None
        matmul = float_3 @ float_4;  float_3 = float_4 = None
        freqs = matmul.transpose(1, 2);  matmul = None
        emb = torch.cat((freqs, freqs), dim = -1);  freqs = None
        cos = emb.cos()
        sin = emb.sin();  emb = None
        _exit_autocast = torch.amp.autocast_mode._exit_autocast(_enter_autocast);  _enter_autocast = _exit_autocast = None
        cos_1 = cos * 1.0;  cos = None
        sin_1 = sin * 1.0;  sin = None
        cos_2 = cos_1.to(dtype = torch.float32);  cos_1 = None
        sin_2 = sin_1.to(dtype = torch.float32);  sin_1 = None
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
        hidden_states = inputs_embeds.to(torch.float32)
        pow_1 = hidden_states.pow(2)
        variance = pow_1.mean(-1, keepdim = True);  pow_1 = None
        add_1 = variance + 1e-06;  variance = None
        rsqrt = torch.rsqrt(add_1);  add_1 = None
        hidden_states_1 = hidden_states * rsqrt;  hidden_states = rsqrt = None
        to_3 = hidden_states_1.to(torch.float32);  hidden_states_1 = None
        hidden_states_2 = l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ * to_3;  l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = to_3 = None
        query_states = torch._C._nn.linear(hidden_states_2, l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_, None);  l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = None
        key_states = torch._C._nn.linear(hidden_states_2, l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_, None);  l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = None
        value_states = torch._C._nn.linear(hidden_states_2, l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_, None);  hidden_states_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = None
        view = query_states.view(2, 8192, 32, 128);  query_states = None
        query_states_1 = view.transpose(1, 2);  view = None
        view_1 = key_states.view(2, 8192, 32, 128);  key_states = None
        key_states_1 = view_1.transpose(1, 2);  view_1 = None
        view_2 = value_states.view(2, 8192, 32, 128);  value_states = None
        value_states_1 = view_2.transpose(1, 2);  view_2 = None
        cos_3 = cos_2.unsqueeze(1);  cos_2 = None
        sin_3 = sin_2.unsqueeze(1);  sin_2 = None
        mul_5 = query_states_1 * cos_3
        x1 = query_states_1[(Ellipsis, slice(None, 64, None))]
        x2 = query_states_1[(Ellipsis, slice(64, None, None))];  query_states_1 = None
        neg = -x2;  x2 = None
        cat_1 = torch.cat((neg, x1), dim = -1);  neg = x1 = None
        mul_6 = cat_1 * sin_3;  cat_1 = None
        q_embed = mul_5 + mul_6;  mul_5 = mul_6 = None
        mul_7 = key_states_1 * cos_3;  cos_3 = None
        x1_1 = key_states_1[(Ellipsis, slice(None, 64, None))]
        x2_1 = key_states_1[(Ellipsis, slice(64, None, None))];  key_states_1 = None
        neg_1 = -x2_1;  x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None
        mul_8 = cat_2 * sin_3;  cat_2 = sin_3 = None
        k_embed = mul_7 + mul_8;  mul_7 = mul_8 = None
        causal_mask_6 = causal_mask_5[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 8192, None))];  causal_mask_5 = None
        query_states_2 = q_embed.contiguous();  q_embed = None
        key_states_2 = k_embed.contiguous();  k_embed = None
        value_states_2 = value_states_1.contiguous();  value_states_1 = None
        attn_output = torch._C._nn.scaled_dot_product_attention(query_states_2, key_states_2, value_states_2, attn_mask = causal_mask_6, dropout_p = 0.0, is_causal = False);  query_states_2 = key_states_2 = value_states_2 = causal_mask_6 = None
        transpose_4 = attn_output.transpose(1, 2);  attn_output = None
        attn_output_1 = transpose_4.contiguous();  transpose_4 = None
        attn_output_2 = attn_output_1.view(2, 8192, -1);  attn_output_1 = None
        attn_output_3 = torch._C._nn.linear(attn_output_2, l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_, None);  attn_output_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = None
        hidden_states_3 = inputs_embeds + attn_output_3;  inputs_embeds = attn_output_3 = None
        hidden_states_4 = hidden_states_3.to(torch.float32)
        pow_2 = hidden_states_4.pow(2)
        variance_1 = pow_2.mean(-1, keepdim = True);  pow_2 = None
        add_5 = variance_1 + 1e-06;  variance_1 = None
        rsqrt_1 = torch.rsqrt(add_5);  add_5 = None
        hidden_states_5 = hidden_states_4 * rsqrt_1;  hidden_states_4 = rsqrt_1 = None
        to_5 = hidden_states_5.to(torch.float32);  hidden_states_5 = None
        hidden_states_6 = l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ * to_5;  l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = to_5 = None
        linear_4 = torch._C._nn.linear(hidden_states_6, l_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_, None);  l_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = None
        silu = torch.nn.functional.silu(linear_4, inplace = False);  linear_4 = None
        linear_5 = torch._C._nn.linear(hidden_states_6, l_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_, None);  hidden_states_6 = l_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = None
        mul_11 = silu * linear_5;  silu = linear_5 = None
        down_proj = torch._C._nn.linear(mul_11, l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_, None);  mul_11 = l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = None
        hidden_states_7 = hidden_states_3 + down_proj;  hidden_states_3 = down_proj = None
        hidden_states_8 = hidden_states_7.to(torch.float32);  hidden_states_7 = None
        pow_3 = hidden_states_8.pow(2)
        variance_2 = pow_3.mean(-1, keepdim = True);  pow_3 = None
        add_7 = variance_2 + 1e-06;  variance_2 = None
        rsqrt_2 = torch.rsqrt(add_7);  add_7 = None
        hidden_states_9 = hidden_states_8 * rsqrt_2;  hidden_states_8 = rsqrt_2 = None
        to_7 = hidden_states_9.to(torch.float32);  hidden_states_9 = None
        hidden_states_10 = l_self_modules_norm_parameters_weight_ * to_7;  l_self_modules_norm_parameters_weight_ = to_7 = None
        return (hidden_states_10,)


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('9feffdf6117a343cadf3331d6af7d459b3e526a1', 131072, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (2, 8192), dtype=torch.int64, is_leaf=True)  # L_input_ids_
    buf1 = reader.storage('d5add94637d92d904e468d5999087df642aa8e52', 524288000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (32000, 4096), requires_grad=True, is_leaf=True)  # L_self_modules_embed_tokens_parameters_weight_
    buf2 = reader.storage('9159cb8bcee7fcb95582f140960cdae72788d326', 2, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf2, (2, 1), dtype=torch.bool, is_leaf=True)  # L_attention_mask_
    buf3 = reader.storage('a872431688a2ef03a882f1260ba338c31ab77c91', 256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (64,), is_leaf=True)  # L_self_modules_rotary_emb_buffers_inv_freq_
    buf4 = reader.storage('c88fd13248fc936f22e7286d9a43f19e92347fbd', 16384, device=device(type='cuda', index=0))
    reader.tensor(buf4, (4096,), requires_grad=True, is_leaf=True)  # L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
    buf5 = reader.storage('4f836cd2c80161cd885616621017e381597433c1', 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf5, (4096, 4096), requires_grad=True, is_leaf=True)  # L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
    buf6 = reader.storage('a125c63eb5226c4482bd5d283c934dc7da30ad91', 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf6, (4096, 4096), requires_grad=True, is_leaf=True)  # L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
    buf7 = reader.storage('c56190a03fea24e470a0cf46bee4cc918957165b', 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf7, (4096, 4096), requires_grad=True, is_leaf=True)  # L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
    buf8 = reader.storage('5da54265c913d9dcd219cad8f8b1475dba171007', 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf8, (4096, 4096), requires_grad=True, is_leaf=True)  # L_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
    buf9 = reader.storage('c88fd13248fc936f22e7286d9a43f19e92347fbd', 16384, device=device(type='cuda', index=0))
    reader.tensor(buf9, (4096,), requires_grad=True, is_leaf=True)  # L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
    buf10 = reader.storage('b039a4866f1189b0742b0a079b54255d49d979b6', 180355072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (11008, 4096), requires_grad=True, is_leaf=True)  # L_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
    buf11 = reader.storage('1653bc5187380407c762f62dd6dc89cf4fca508e', 180355072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (11008, 4096), requires_grad=True, is_leaf=True)  # L_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
    buf12 = reader.storage('702097726d4b8d84b98aa42997bba533d734a7e8', 180355072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (4096, 11008), requires_grad=True, is_leaf=True)  # L_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
    buf13 = reader.storage('c88fd13248fc936f22e7286d9a43f19e92347fbd', 16384, device=device(type='cuda', index=0))
    reader.tensor(buf13, (4096,), requires_grad=True, is_leaf=True)  # L_self_modules_norm_parameters_weight_
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='minify', autocast=False, backend='eager')

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config






isolate_fails_code_str = None



# torch version: 2.6.0.dev20241030+cu124
# torch cuda version: 12.4
# torch git version: e47e8794499a4a0130ff4efb8713ff93f4b40c36


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Thu_Mar_28_02:18:24_PDT_2024 
# Cuda compilation tools, release 12.4, V12.4.131 
# Build cuda_12.4.r12.4/compiler.34097967_0 

# GPU Hardware Info: 
# NVIDIA H100 80GB HBM3 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15):
        embedding = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = None
        iota = torch.ops.prims.iota.default(8192, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(iota, 0)
        full_default = torch.ops.aten.full.default([8192, 1], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_1 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(iota_1, -2)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(iota, -1)
        sub = torch.ops.aten.sub.Tensor(unsqueeze_1, unsqueeze_2);  unsqueeze_1 = unsqueeze_2 = None
        ge = torch.ops.aten.ge.Scalar(sub, 1);  sub = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ge, full_default, full_default_1);  ge = full_default = full_default_1 = None
        view = torch.ops.aten.view.default(iota, [-1, 1]);  iota = None
        gt = torch.ops.aten.gt.Tensor(iota_1, view);  iota_1 = view = None
        mul = torch.ops.aten.mul.Tensor(where, gt);  where = gt = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(mul, 0);  mul = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1);  unsqueeze_5 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_6, [2, 1, -1, -1]);  unsqueeze_6 = None
        clone = torch.ops.aten.clone.default(expand_1);  expand_1 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(primals_3, 1);  primals_3 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        add = torch.ops.aten.add.Tensor(clone, unsqueeze_8);  unsqueeze_8 = None
        eq = torch.ops.aten.eq.Scalar(add, 0);  add = None
        full_default_2 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(eq, full_default_2, clone);  eq = full_default_2 = clone = None
        eq_1 = torch.ops.aten.eq.Scalar(where_1, -3.4028234663852886e+38)
        logical_not = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_1 = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1 = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        bitwise_not = torch.ops.aten.bitwise_not.default(logical_not_1);  logical_not_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(where_1, bitwise_not);  where_1 = bitwise_not = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(primals_4, 0);  primals_4 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_10, [1, -1, 1]);  unsqueeze_10 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        convert_element_type = torch.ops.prims.convert_element_type.default(unsqueeze_11, torch.float32);  unsqueeze_11 = None
        expand_3 = torch.ops.aten.expand.default(expand_2, [1, 64, 1]);  expand_2 = None
        expand_4 = torch.ops.aten.expand.default(convert_element_type, [1, 1, 8192]);  convert_element_type = None
        bmm = torch.ops.aten.bmm.default(expand_3, expand_4);  expand_3 = expand_4 = None
        permute = torch.ops.aten.permute.default(bmm, [0, 2, 1])
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_5 = torch.ops.aten.expand.default(unsqueeze_12, [1, 8192, 2, 64]);  unsqueeze_12 = None
        clone_1 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_4 = torch.ops.aten.view.default(clone_1, [1, 8192, 128]);  clone_1 = None
        cos = torch.ops.aten.cos.default(view_4)
        sin = torch.ops.aten.sin.default(view_4);  view_4 = None
        mul_2 = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        mul_3 = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(embedding, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1 = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(embedding, rsqrt)
        mul_5 = torch.ops.aten.mul.Tensor(primals_5, mul_4);  mul_4 = None
        permute_1 = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        view_5 = torch.ops.aten.view.default(mul_5, [16384, 4096]);  mul_5 = None
        mm = torch.ops.aten.mm.default(view_5, permute_1)
        view_6 = torch.ops.aten.view.default(mm, [2, 8192, 4096]);  mm = None
        permute_2 = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        mm_1 = torch.ops.aten.mm.default(view_5, permute_2)
        view_8 = torch.ops.aten.view.default(mm_1, [2, 8192, 4096]);  mm_1 = None
        permute_3 = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        mm_2 = torch.ops.aten.mm.default(view_5, permute_3)
        view_10 = torch.ops.aten.view.default(mm_2, [2, 8192, 4096]);  mm_2 = None
        view_11 = torch.ops.aten.view.default(view_6, [2, 8192, 32, 128]);  view_6 = None
        permute_4 = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        view_12 = torch.ops.aten.view.default(view_8, [2, 8192, 32, 128]);  view_8 = None
        permute_5 = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        view_13 = torch.ops.aten.view.default(view_10, [2, 8192, 32, 128]);  view_10 = None
        permute_6 = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(mul_2, 1);  mul_2 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(mul_3, 1);  mul_3 = None
        mul_6 = torch.ops.aten.mul.Tensor(permute_4, unsqueeze_13)
        slice_24 = torch.ops.aten.slice.Tensor(permute_4, 3, 0, 64)
        slice_25 = torch.ops.aten.slice.Tensor(permute_4, 3, 64, 9223372036854775807);  permute_4 = None
        neg = torch.ops.aten.neg.default(slice_25);  slice_25 = None
        cat = torch.ops.aten.cat.default([neg, slice_24], -1);  neg = slice_24 = None
        mul_7 = torch.ops.aten.mul.Tensor(cat, unsqueeze_14);  cat = None
        add_2 = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(permute_5, unsqueeze_13);  unsqueeze_13 = None
        slice_26 = torch.ops.aten.slice.Tensor(permute_5, 3, 0, 64)
        slice_27 = torch.ops.aten.slice.Tensor(permute_5, 3, 64, 9223372036854775807);  permute_5 = None
        neg_1 = torch.ops.aten.neg.default(slice_27);  slice_27 = None
        cat_1 = torch.ops.aten.cat.default([neg_1, slice_26], -1);  neg_1 = slice_26 = None
        mul_9 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_14);  cat_1 = unsqueeze_14 = None
        add_3 = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        slice_31 = torch.ops.aten.slice.Tensor(mul_1, 3, 0, 8192);  mul_1 = None
        clone_3 = torch.ops.aten.clone.default(add_2, memory_format = torch.contiguous_format);  add_2 = None
        clone_4 = torch.ops.aten.clone.default(add_3, memory_format = torch.contiguous_format);  add_3 = None
        clone_5 = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        constant_pad_nd = torch.ops.aten.constant_pad_nd.default(slice_31, [0, 7], 0.0);  slice_31 = None
        slice_32 = torch.ops.aten.slice.Tensor(constant_pad_nd, -1, 0, 1);  constant_pad_nd = None
        expand_6 = torch.ops.aten.expand.default(slice_32, [2, 32, 8192, 8192])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_3, clone_4, clone_5, expand_6, True);  expand_6 = None
        getitem = _scaled_dot_product_efficient_attention[0]
        getitem_1 = _scaled_dot_product_efficient_attention[1]
        getitem_2 = _scaled_dot_product_efficient_attention[2]
        getitem_3 = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
        permute_7 = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3])
        view_14 = torch.ops.aten.view.default(permute_7, [2, 8192, -1]);  permute_7 = None
        permute_8 = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        view_15 = torch.ops.aten.view.default(view_14, [16384, 4096]);  view_14 = None
        mm_3 = torch.ops.aten.mm.default(view_15, permute_8);  view_15 = None
        view_16 = torch.ops.aten.view.default(mm_3, [2, 8192, 4096])
        add_4 = torch.ops.aten.add.Tensor(embedding, view_16);  view_16 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(add_4, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_5 = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_10 = torch.ops.aten.mul.Tensor(add_4, rsqrt_1)
        mul_11 = torch.ops.aten.mul.Tensor(primals_10, mul_10);  mul_10 = None
        permute_9 = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        view_17 = torch.ops.aten.view.default(mul_11, [16384, 4096]);  mul_11 = None
        mm_4 = torch.ops.aten.mm.default(view_17, permute_9)
        view_18 = torch.ops.aten.view.default(mm_4, [2, 8192, 11008])
        sigmoid = torch.ops.aten.sigmoid.default(view_18)
        mul_12 = torch.ops.aten.mul.Tensor(view_18, sigmoid);  view_18 = sigmoid = None
        permute_10 = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        mm_5 = torch.ops.aten.mm.default(view_17, permute_10)
        view_20 = torch.ops.aten.view.default(mm_5, [2, 8192, 11008])
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, view_20);  mul_12 = view_20 = None
        permute_11 = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        view_21 = torch.ops.aten.view.default(mul_13, [16384, 11008]);  mul_13 = None
        mm_6 = torch.ops.aten.mm.default(view_21, permute_11)
        view_22 = torch.ops.aten.view.default(mm_6, [2, 8192, 4096]);  mm_6 = None
        add_6 = torch.ops.aten.add.Tensor(add_4, view_22);  add_4 = view_22 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
        mean_2 = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_7 = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_14 = torch.ops.aten.mul.Tensor(add_6, rsqrt_2)
        mul_15 = torch.ops.aten.mul.Tensor(primals_14, mul_14);  mul_14 = None
        permute_12 = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        view_23 = torch.ops.aten.view.default(mul_15, [16384, 4096]);  mul_15 = None
        mm_7 = torch.ops.aten.mm.default(view_23, permute_12)
        view_24 = torch.ops.aten.view.default(mm_7, [2, 8192, 32000]);  mm_7 = None
        permute_15 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_19 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_23 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_28 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_32 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_40 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_44 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_48 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (view_24, primals_1, primals_5, primals_10, primals_14, embedding, bmm, rsqrt, view_5, clone_3, clone_4, clone_5, slice_32, getitem, getitem_1, getitem_2, getitem_3, mm_3, rsqrt_1, view_17, mm_4, mm_5, view_21, add_6, rsqrt_2, view_23, permute_15, permute_19, permute_23, permute_28, permute_32, permute_40, permute_44, permute_48)
        
def load_args(reader):
    buf0 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (2, 8192), dtype=torch.int64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 524288000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (32000, 4096), requires_grad=True, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 2, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf2, (2, 1), dtype=torch.bool, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (64,), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf4, (4096,), requires_grad=True, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf5, (4096, 4096), requires_grad=True, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf6, (4096, 4096), requires_grad=True, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf7, (4096, 4096), requires_grad=True, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf8, (4096, 4096), requires_grad=True, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf9, (4096,), requires_grad=True, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 180355072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (11008, 4096), requires_grad=True, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 180355072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (11008, 4096), requires_grad=True, is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 180355072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (4096, 11008), requires_grad=True, is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf13, (4096,), requires_grad=True, is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 524288000, device=device(type='cuda', index=0))
    reader.tensor(buf14, (32000, 4096), requires_grad=True, is_leaf=True)  # primals_15
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='minify', tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', tracing_mode='real', check_str=None)
        # mod(*args)