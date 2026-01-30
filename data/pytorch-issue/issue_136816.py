Python
import torch
import torch.nn as nn

mha=nn.MultiheadAttention(embed_dim=1, num_heads=1, dropout=0, bias=False, add_bias_kv=False,
                          add_zero_attn=False, kdim=None, vdim=None, batch_first=True,
                          device=None)
mha.in_proj_weight.data =torch.tensor([[1],[3],[2]],dtype=torch.float)
mha.out_proj.weight.data=torch.tensor([[2]],dtype=torch.float)
inputdata=torch.tensor([[[1.],[2.]]])
out=mha(inputdata,inputdata,inputdata,average_attn_weights=False,need_weights=True,
        is_causal=None,attn_mask=None,key_padding_mask=torch.tensor([[0.,1]],dtype=torch.float))#issue is over here "key_padding_mask=torch.tensor([[0.,1]],dtype=torch.float)" which is wrong output,however key_padding_mask=torch.tensor([[1.,0]],dtype=torch.float) is normal output as described in api. 
print('inputdata:',inputdata,'in_proj_weight:',mha.in_proj_weight,
      'out_proj.weight:',mha.out_proj.weight)
print(out)

import torch
import torch.nn as nn

mha = nn.MultiheadAttention(
    embed_dim=1,
    num_heads=1,
    dropout=0,
    bias=False,
    add_bias_kv=False,
    add_zero_attn=False,
    kdim=None,
    vdim=None,
    batch_first=True,
    device=None,
)

mha.in_proj_weight.data = torch.tensor([[1], [3], [2]], dtype=torch.float)
mha.out_proj.weight.data = torch.tensor([[2]], dtype=torch.float)

input_data = torch.tensor([[[1.0], [2.0]]])
key_padding_mask = torch.tensor([[False, True]], dtype=torch.bool)
key_padding_mask = torch.tensor([[0.,1]],dtype=torch.float)
# key_padding_mask = torch.tensor([[1,0.]],dtype=torch.float)

out, attn_weights = mha(
    input_data,
    input_data,
    input_data,
    average_attn_weights=False,
    need_weights=True,
    is_causal=None,
    attn_mask=None,
    key_padding_mask=key_padding_mask,
)

print("Input data:", input_data)
print("in_proj_weight:", mha.in_proj_weight)
print("out_proj.weight:\n", mha.out_proj.weight)
print("Output:\n", out)
print("Attention weights:\n", attn_weights)
print("Attention weights sum:\n", torch.sum(attn_weights, dim=-1))