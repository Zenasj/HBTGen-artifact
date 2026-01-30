import torch
import torch.nn as nn
from torch import Tensor

import logging
# torch._logging.set_logs(guards=True)
torch._logging.set_logs(recompiles=True, recompiles_verbose = True, fusion = True)

Mod = nn.Module

class A(Mod):
    def __init__(self, v:float):
        super().__init__()
        self.b = v
        
    def forward(self, x):
        return (x.sin()+self.b).sin()
       
def forward(net, x:Tensor) -> Tensor:
    return net.forward(x).sum()


dev = torch.device('cuda:0')

compile_args = dict(fullgraph=True, dynamic = True, backend = "inductor", options={'force_same_precision':True, 'disable_cpp_codegen':False, 'trace.graph_diagram':True, "triton.cudagraphs": False})
cforward = torch.compile(forward, **compile_args)

def compute(net):
    y = cforward(net, x)
    y.backward()

start_e = torch.cuda.Event(enable_timing=True)
end_e = torch.cuda.Event(enable_timing=True)
x = torch.rand((100,100,100), device =dev, requires_grad=True)
# for d in[1,2]:
#     torch._dynamo.mark_dynamic(x, d)

nets = [A(float(1.0)) for j in range(3)]
# nets = [nn.Sequential(*[A(float(j)) for j in range(3)]) for i in range(3)]

# WARMUP/ COMPILE
s = torch.cuda.Stream(device=dev)
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.device(dev):
    with torch.cuda.stream(s):  # warmup on a side stream, according to examples
        # check correctness
        for i in range(3):
            y = forward(nets[i], x)
            y.backward()
        # warmup compiled
        for i in range(3):
            print(f"_______________Net {i}_______________")    
            x.grad = None
            start_e.record()
            compute(nets[i])
            end_e.record()
            torch.cuda.synchronize()
            t = start_e.elapsed_time(end_e)
            print(f'Time: {t} ms')
            print(x.grad.mean())
            
    torch.cuda.current_stream().wait_stream(s)
    G = torch.cuda.CUDAGraph()
    with torch.cuda.graph(G):
        x.grad = None
        for i in range(3):
            compute(nets[i])

    for i in range(3):
        print(f"_______________Compiled Graphed Iteration {i}_______________")    
        tt = 0
        for j in range(100):
            start_e.record()
            G.replay()
            end_e.record()
            torch.cuda.synchronize()
            t = start_e.elapsed_time(end_e)
            tt += t
        print(f'Time: {tt/100} ms')
        print(x.grad.mean())

from importlib import reload
def get_Mod():
    mod = reload(torch.nn.modules.module)
    return mod.Module
Mod = get_Mod()