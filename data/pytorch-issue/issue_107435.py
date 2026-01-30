import torch.nn as nn

class MyLayer(torch.nn.Module):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.mlp_up_proj = nn.Linear(args.hs, 4 * args.hs)
        self.mlp_act = nn.GELU(approximate='none')
        self.mlp_down_proj = nn.Linear(4*args.hs, args.hs)
        
    def forward(self, x):
        y1 = self.mlp_up_proj(x)
        y2 = self.mlp_act(y1)
        y3 = self.mlp_down_proj(y2)
        return y3

# it does not work
class MyLayerImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mlp_up_proj, mlp_act, mlp_down_proj):
        with torch.enable_grad():
            y1 = mlp_up_proj(x)
            y2 = mlp_act(y1)
            y3 = mlp_down_proj(y2)
        y4 = y3.detach()
        ctx.save_for_backward(x, y1, y2, y3)
        return y4

    @staticmethod
    def backward(ctx, dout, *args):
        x, y1, y2, y3 = ctx.saved_tensors
        din = y3.backward(dout)
        return din, None, None, None
    

class MyLayer(torch.nn.Module):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.mlp_up_proj = nn.Linear(args.hs, 4 * args.hs)
        self.mlp_act = nn.GELU(approximate='none')
        self.mlp_down_proj = nn.Linear(4*args.hs, args.hs)
        
    def forward(self, x):
        return MyLayerImpl.apply(x, self.mlp_up_proj, self.mlp_act, self.mlp_down_proj)

import torch
import argparse
from torch import nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(0)

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('--mbs', type=int, default=2)
parser.add_argument('--seq', type=int, default=8)
parser.add_argument('--hs', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()


for k, v in vars(args).items():
    print(k, '=', v)

    
class MyLayer(torch.nn.Module):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.mlp_up_proj = nn.Linear(args.hs, 4 * args.hs)
        self.mlp_down_proj = nn.Linear(4*args.hs, args.hs)
        
    def forward(self, x):
        with record_function("xxxxmylayer:forward"):
            y = self.mlp_up_proj(x)
            y = self.mlp_down_proj(y)
            return y


class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.Wqkv = nn.Linear(args.hs, 3 * args.hs)
        self.out_proj = nn.Linear(args.hs, args.hs)
        self.my_layer1 = MyLayer()
        self.my_layer2 = MyLayer()
        
    def forward(self, x):
        qkv = self.Wqkv(x)
        query, key, value = qkv.chunk(3, dim=2)
        new_v = query + key + value
        proj = self.out_proj(new_v)
        proj = self.my_layer1(proj)
        proj = self.my_layer2(proj)
        return proj


def main():
    model = TestNet()
    model.bfloat16().to(args.device)

    x = torch.randn([args.mbs, args.seq, args.hs]).bfloat16().to(args.device)
    x.requires_grad_()
    
    # warm up
    for i in range(3):
        y = model(x)
        loss = y.sum()
        loss.backward()
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 with_stack=True,
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./timeline', worker_name='worker0')) as prof:
        y = model(x)
        loss = y.sum()
        loss.backward()

    
if __name__ == "__main__":
    main()