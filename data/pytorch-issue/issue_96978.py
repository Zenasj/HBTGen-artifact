import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, out):
        out = self.softmax(out)    
        out = F.dropout(out, p=0.1, training=True)

        return out

def generate_io_tensor(net):
    input_tensors = []
    for idx, shape in enumerate(input_shapes):
        tensor = torch.rand(shape, dtype=torch.float16, requires_grad=True, device='cuda')
        input_tensors.append(tensor)
    
    target_tensor = net(*input_tensors)
    
    return input_tensors, target_tensor

config = {'seq_len': 8192, 'num_attn_heads':16, 'batch_size': 4} #Fails
#config = {'seq_len': 8192, 'num_attn_heads':16, 'batch_size': 2} #Works

net = Net(config)
net.cuda()
net = net.half()
                   
network_fn = torch.compile(net)

bench_iters = 5
input_shapes = [(config['batch_size'], config['num_attn_heads'], config['seq_len'], config['seq_len'])]

for idx in range(bench_iters):

    input_tensors, target_tensor = generate_io_tensor(net)
    for inp_tensor in input_tensors:
        inp_tensor.grad = None
    
    outputs = network_fn(*input_tensors)
    outputs.backward(target_tensor)