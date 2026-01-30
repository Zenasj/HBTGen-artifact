import torch
import torch.nn as nn

class myModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("test", torch.as_tensor(1.0))
        self.which_foo = self.foo
        
    def forward(self, x):
        return x

    def foo(self, x):
        return

for _ in range(3):
    m = myModule().cuda()
    
    del m
    torch.cuda.empty_cache()
    print("\n")
    print("mem alloc:", torch.cuda.memory_allocated())
    print("max mem alloc:", torch.cuda.max_memory_allocated())