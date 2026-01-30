import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            1, 2, kernel_size=(509,2), stride=3, padding=255, dilation=(1, 1014),
        )
        
    def forward(self, i0, i1):
        x = torch.max(i0, i1)
        y = self.conv1(x)
        return y

i0 = torch.zeros((1,1,2,505), dtype=torch.float32)
i1 = torch.zeros((1,2,505), dtype=torch.float32)

mod = MyModule()

out = mod(i0, i1)
print(f'eager: out = {out}') # <-- works fine

exported = torch.jit.trace(mod, [i0, i1])
exported = torch.jit.optimize_for_inference(exported) # <-- RuntimeError: could not construct a memory descriptor using a format tag

eout = exported(i0, i1)
print(f'JIT: eout = {eout}')

assert torch.allclose(out, eout)