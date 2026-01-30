import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(10, 20)
        self.linear2 = torch.nn.Linear(20, 30)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.cat((x, x), dim=1)
        x = x.view(-1, 2, 30)
        x = x[:, 1, :]
        x = self.relu(x)
        return x

device = 'cuda'
batch_size = 2
x = torch.randn(batch_size, 10).to(device)
func = Model().to(device)

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)

    res1 = func(x) # without jit
    print(res1)
    # succeed

    res2 = jit_func(x)
    # /_inductor/pattern_matcher.py", line 869, in stable_topological_sort
    # assert not waiting and len(ready) == len(graph.nodes)
    # torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
    # AssertionError: