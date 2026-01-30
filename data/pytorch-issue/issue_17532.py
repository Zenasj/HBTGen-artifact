import torch
import torch.jit
from typing import List


def fn1(x):
    # type: (List[Tensor]) -> List[Tensor]
    out = []
    for t in x:
       out.append(t*2 + t*t)
    return out

def traceandprint(f, inputs):
   traced_fn = torch.jit.script(f, inputs)
   out = traced_fn(inputs)
   print(traced_fn.graph_for(inputs))
   return traced_fn, out

x=torch.ones(5,5, device="cuda", requires_grad = False)
y=torch.ones(5,5, device="cuda", requires_grad = False )

fn, out  = traceandprint(fn1, [x,y])