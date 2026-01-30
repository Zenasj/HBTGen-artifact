import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Net(nn.Module):

    def __init__(self, config):
        super().__init__()
        
    def forward(self, q, k, v):
        attn_weight = torch.softmax(
            (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))),
            dim=-1,
        )
        return attn_weight @ v

class Net_w_dropout(nn.Module):

    def __init__(self, config):
        super().__init__()
        
    def forward(self, q, k, v):
        
        attn_weight = torch.softmax(
            (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))),
            dim=-1,
        )
        attn_weight = torch.dropout(attn_weight, 0.1, True)

        return attn_weight @ v

class Net_w_mask(nn.Module):

    def __init__(self, config):
        super().__init__()
        
    def forward(self, q, k, v):

        attn_mask = torch.ones(
            q.size(-2), k.size(-2), dtype=torch.bool, device=q.device
        ).tril(diagonal=0)
        
        attn_mask = attn_mask.masked_fill(
            torch.logical_not(attn_mask), -float("inf")
        )
        
        attn_weight = torch.softmax(
            (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask,
            dim=-1,
        )
        
        return attn_weight @ v

def generate_io_tensor():
    input_tensors = []
    for idx, shape in enumerate(input_shapes):
        tensor = torch.rand(shape, dtype=torch.float32, requires_grad=True, device='cuda')
        input_tensors.append(tensor)
    
    return input_tensors

config = {'seq_len': 2048, 'num_attn_heads':16, 'batch_size': 4, 'kv_channels': 64}

input_shapes = [(config['batch_size'], config['num_attn_heads'], config['seq_len'], config['kv_channels']),
                (config['batch_size'], config['num_attn_heads'], config['seq_len'], config['kv_channels']),
                (config['batch_size'], config['num_attn_heads'], config['seq_len'], config['kv_channels'])]


input_tensors = generate_io_tensor()
for inp_tensor in input_tensors:
    inp_tensor.grad = None

net = Net(config)
net.cuda()
network_fn = torch.compile(net, fullgraph=True)

from torch._inductor.utils import run_and_get_code
result, (source_code,) = run_and_get_code(
        network_fn, *input_tensors)

print(source_code)

net_w_dropout = Net_w_dropout(config)
net_w_dropout.cuda()
network_fn_dropout = torch.compile(net_w_dropout, fullgraph=True)

from torch._inductor.utils import run_and_get_code
result, (source_code,) = run_and_get_code(
        network_fn_dropout, *input_tensors)

print(source_code)

net_w_mask = Net_w_mask(config)
net_w_mask.cuda()
network_fn_mask = torch.compile(net_w_mask, fullgraph=True)

from torch._inductor.utils import run_and_get_code
result, (source_code,) = run_and_get_code(
        network_fn_mask, *input_tensors)

print(source_code)