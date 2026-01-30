import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 3)
        self.linear3 = torch.nn.Linear(3, 2)

    def forward(self, x1):
        return torch.nn.functional.dropout(torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias), p=0.8).argmax(dim=-1).repeat(1, 3).add_(1)

func = Model().to('cuda')

x = torch.randn(1, 2).cuda()

with torch.no_grad():
    func = func.eval()
    res1 = func(x) # without jit
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    # torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
    # RuntimeError: PassManager::run failed