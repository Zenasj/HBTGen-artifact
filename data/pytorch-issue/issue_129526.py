import torch
import torch.nn as nn

@torch.compiler.allow_in_graph
def composite_op(x):
    if x.sum() > 0:
        return torch.sigmoid(x)
    else:
        return torch.relu(x)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = composite_op(x)
        return x

graph_cnt = 0
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    global graph_cnt
    print(f"============ graph {graph_cnt} ============")
    graph_cnt += 1
    print(gm)
    return gm  # return a python callable


toy = Toy()
compiled_mod = torch.compile(toy, backend=my_compiler)
y = compiled_mod(torch.randn(12, 10))

def composite_op_fakemode_fallback(x):
    return x

@torch.compiler.allow_in_graph(fakemode_fallback_fn=composite_op_fakemode_fallback)
def composite_op(x):
    if x.sum() > 0:
        return torch.sigmoid(x)
    else:
        return torch.relu(x)