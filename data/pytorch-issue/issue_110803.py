import torch
import torch.nn as nn
from torch._dynamo.utils import CompileProfiler

class SimpleDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.dropout(self.linear(x))
    
model = SimpleDropout()
x = torch.randn(10)
with CompileProfiler() as p:
    model = torch.compile(model, backend=p)
    for _ in range(4):
        model.eval()
        model(x)
        model.train()
        model(x)
    print(p.report())