import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(config['input_units'], config['hidden_units'], bias=True)
        
    def forward(self, x):
        out      = self.fc1(x)
        
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = {'input_units': 768, 'hidden_units': 768, 'seq_len': 512, 'batch_size': 64}

net = Net(config)
net.cuda()
net = net.half()

input_shape    = [config['seq_len'], config['batch_size'], config['input_units']]
target_shape   = [config['seq_len'], config['batch_size'], config['hidden_units']]

network_fn = torch.compile(net, mode="max-autotune")
#network_fn = torch.compile(net)

bench_iters = 8
profile_batch = 5

for idx in range(bench_iters):

    inp_tensor = torch.rand(input_shape, dtype=torch.float16, requires_grad=True, device='cuda')
    tgt_tensor = torch.rand(target_shape, dtype=torch.float16, requires_grad=True, device='cuda')
    
    outputs = network_fn(inp_tensor)
    outputs.backward(tgt_tensor)