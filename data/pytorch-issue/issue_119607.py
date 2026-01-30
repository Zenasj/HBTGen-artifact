import torch.nn as nn

import gc
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # self.fc1 = torch.nn.Linear(3000, 50000)
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(3000, 10000),
            torch.nn.ReLU(),
            torch.nn.Linear(10000, 50000),
            torch.nn.ReLU(),
            torch.nn.Linear(50000, 20000),
            torch.nn.ReLU(),
            torch.nn.Linear(20000, 1234),
        )

    def forward(self, out):
        out = self.fc1(out)
        return out

def run(compile):
    mod = MyModel().cuda()
    if compile:
        mod = torch.compile(mod, backend="eager")
    inp = torch.rand(10000, 3000).cuda()
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

import gc
import weakref
import torch

mod = torch.nn.Linear(3000, 50000).cuda()
def fn(x):
    return mod(x)

ref = weakref.ref(mod, lambda _: print("mod deleted"))
weakref.finalize(fn, lambda: print("fn deleted"))

inp = torch.rand(10000, 3000).cuda()

torch.compile(backend="eager")(fn)(inp)

del mod
del fn

gc.collect()

# expect finalizers to run before this point
breakpoint()