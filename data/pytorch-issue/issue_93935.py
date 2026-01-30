import torch
import torch.nn as nn
import torch._dynamo as dynamo
from torch._dynamo import disallow_in_graph
import random

# manually simulate graph breaks
def graph_break():
    pass
disallow_in_graph(graph_break)

class Repro(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 4)
        self.linear3 = nn.Linear(4, 4)
    def forward(self, x):
        x = self.linear1(x)
        graph_break()
        x = self.linear2(x)
        graph_break()
        x = self.linear3(x)
        return x

if __name__ == '__main__':
    model = Repro().cuda()
    opt = torch.compile(model, backend='aot_eager')
    # variable batch_size to simulate recompilations
    for batch_size in range(3):
        data = torch.randn(batch_size, 4).cuda()
        out = opt(data)
        loss = out.sum()
        loss.backward()

def _wrapped_bw_compiler(*args, **kwargs):
    global _count
    _count += 1
    print(f'{_count}: {bw_compiler}')
    return eval_frame.disable(eval_frame.disable(bw_compiler)(*args, **kwargs))

import torch
import torch.nn as nn
import torch._dynamo.config
from torch._dynamo import disallow_in_graph
import sys

# 6 graph breaks x 50 recompilations, exceeds 300 recursion limit
sys.setrecursionlimit(300)

# manually simulate graph breaks
def graph_break():
    pass
disallow_in_graph(graph_break)

# A model with 6 graph breaks
class Repro(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 4)
        self.linear3 = nn.Linear(4, 4)
        self.linear4 = nn.Linear(4, 4)
        self.linear5 = nn.Linear(4, 4)
        self.linear6 = nn.Linear(4, 4)
    def forward(self, x):
        x = self.linear1(x)
        graph_break()
        x = self.linear2(x)
        graph_break()
        x = self.linear3(x)
        graph_break()
        x = self.linear4(x)
        graph_break()
        x = self.linear5(x)
        graph_break()
        x = self.linear6(x)
        return x

if __name__ == '__main__':
    model = Repro().cuda()
    opt = torch.compile(model, backend='aot_eager')
    # opt = torch.compile(model, backend='inductor')  # inductor gives no error
    # variable batch_size to simulate recompilations
    for batch_size in range(50):
        data = torch.randn(batch_size, 4).cuda()
        out = opt(data)
        loss = out.sum()
        loss.backward()