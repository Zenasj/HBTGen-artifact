import torch.nn as nn

import torch
from torch.utils.checkpoint import checkpoint

def report_memory(name):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {:.1f}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {:.1f}".format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += " | reserved: {:.1f}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {:.1f}".format(torch.cuda.max_memory_reserved() / mega_bytes)
    print(string)

class TestNN(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ln1 = torch.nn.Linear(10,20)
        self.ln2 = torch.nn.Linear(20,3000)
        self.ln3 = torch.nn.Linear(3000,40)
        
    def forward(self, x):
        x = self.ln1(x)
        x = torch.nn.functional.relu(x)
        # intend to use inner saved_tensors_hooks to disable _checkpoint_hook
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            x = self.ln2(x)
            x = torch.nn.functional.relu(x)
        x = self.ln3(x)
        x = torch.nn.functional.relu(x)
        return x
    
model = TestNN().cuda()
optim = torch.optim.AdamW(model.parameters())

for _ in range(100):
    x = torch.randn(4, 10).cuda()
    out = checkpoint(model, x, use_reentrant=False)
    (out - torch.randn_like(out)).sum().backward()
    optim.step()
    optim.zero_grad()
    report_memory("Mem")