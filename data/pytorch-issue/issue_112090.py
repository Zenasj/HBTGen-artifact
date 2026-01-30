import torch.nn as nn

import gc
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(40000, 10000)

    def forward(self, out):
        out = self.fc1(out)
        return out

def run(compile):
    mod = MyModel().cuda()
    if compile:
        mod = torch.compile(mod, backend="eager")
    inp = torch.rand(10000, 40000).cuda()
    mod(inp)

def clean_and_report_memory():
    gc.collect()
    print(f"max memory: {torch.cuda.max_memory_allocated()}, curr memory: {torch.cuda.memory_allocated()}")

run(False)
clean_and_report_memory()

run(True)
clean_and_report_memory()

torch._dynamo.reset()
clean_and_report_memory()