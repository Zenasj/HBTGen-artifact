from typing import List
from torch._dynamo import optimize
import torch._dynamo as dynamo
import torch

dynamo.config.output_code = True
torch._dynamo.reset()

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(gm.code)
    print(gm.graph)
    gm.graph.print_tabular()
    return gm.forward

s = torch.cuda.Stream()

@dynamo.optimize(my_compiler)
def fn(t) -> torch.Tensor:
    tmp1 = torch.mul(t, 5)
    tmp2 = torch.add(tmp1, 2)
    with torch.cuda.stream(s):
      r = torch.relu(tmp2)
    s1 = torch.add(r, 2)
    s2 = torch.cos(s1)
    return s2


i = torch.Tensor([-2, 3]).to('cuda')
r = fn(i)
print(f"r = {r}")