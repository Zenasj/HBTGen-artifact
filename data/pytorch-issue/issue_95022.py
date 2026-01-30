import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch._inductor import config
config.comment_origin = True

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(config['input_units'], config['hidden_units'], bias=True)
        self.ln = nn.LayerNorm(config['hidden_units'])
        self.fc2 = nn.Linear(config['hidden_units'], config['output_units'], bias=False)
        
    def forward(self, x, residuals):
        out      = self.fc1(x)
        out      = F.dropout(out, p=0.1, training=True)
        ln_input = out + residuals
        ln_out   = self.ln(ln_input)
        out      = self.fc2(ln_out)

        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = {'input_units': 768, 'hidden_units': 768, 'seq_len': 512, 'batch_size': 64, 'output_units': 2048}

net = Net(config)
net.cuda()
net = net.half()

input_shape    = [config['seq_len'], config['batch_size'], config['input_units']]
residual_shape = [config['seq_len'], config['batch_size'], config['hidden_units']]
target_shape   = [config['seq_len'], config['batch_size'], config['output_units']]

network_fn = torch.compile(net)

bench_iters = 10

for idx in range(bench_iters):

    inp_tensor = torch.rand(input_shape, dtype=torch.float16, requires_grad=True, device='cuda')
    res_tensor = torch.rand(residual_shape, dtype=torch.float16, requires_grad=True, device='cuda')
    tgt_tensor = torch.rand(target_shape, dtype=torch.float16, requires_grad=True, device='cuda')

    outputs = network_fn(inp_tensor, res_tensor)
    outputs.backward(tgt_tensor)