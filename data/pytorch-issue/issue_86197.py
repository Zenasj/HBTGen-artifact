import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchdynamo

tensor_dtype = torch.float16

ap = argparse.ArgumentParser()
ap.add_argument("-td", "--torchdynamo", action='store_true', default=False, help='Enable Nvfuser + TorchDynamo optimization pass')

args = vars(ap.parse_args())
torchdynamo_enable = bool(args['torchdynamo'])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.scale = 8
        self.mask = torch.tril(torch.ones([16, 1, 2048, 2048], dtype=torch.float16, device='cuda'))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, attention_scores):
        #Scale-Mask-Softmax-Dropout
        out = attention_scores * self.scale
        out = out + ((1.0 - self.mask) * -10000.0)
        #out = F.gelu(out)
        out = self.softmax(out)
        out = F.dropout(out, p=0.1, training=True)

        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
net = net.cuda()
net = net.half()

if torchdynamo_enable:
    network_fn = torchdynamo.optimize("aot_nvfuser")(net)
else:
    network_fn = torch.jit.script(net)

shape = [16, 16, 2048, 2048]
q = torch.rand(shape, dtype=tensor_dtype, requires_grad=True, device=device)
t = torch.rand(shape, dtype=tensor_dtype, requires_grad=True, device=device)

outputs = network_fn(q)
outputs.backward(t)

import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.scale = 8
        self.mask = torch.tril(torch.ones([16, 1, 2048, 2048], dtype=torch.float16, device='cuda'))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, attention_scores):
        #Scale-Mask-Softmax-Dropout
        out = attention_scores * self.scale
        out = out + ((1.0 - self.mask) * -10000.0)
        #out = F.gelu(out)
        out = self.softmax(out)
        out = F.dropout(out, p=0.1, training=True)

        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tensor_dtype = torch.float16

net = Net()
net = net.cuda()
net = net.half()

network_fn = torchdynamo.optimize("aot_nvfuser")(net)

shape = [16, 16, 2048, 2048]
q = torch.rand(shape, dtype=tensor_dtype, requires_grad=True, device=device)
t = torch.rand(shape, dtype=tensor_dtype, requires_grad=True, device=device)

def bench(f):
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    begin = time.time()
    for _ in range(100):
        f()
    torch.cuda.synchronize()
    print((time.time()-begin)*1e6/100)

bench(lambda: network_fn(q).backward(t))

Before: 26246.845722198486
After: 21274.752616882324