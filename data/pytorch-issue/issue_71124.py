import torch.nn as nn

import torch
from torch.utils.checkpoint import checkpoint

def main():
    class Linear(torch.nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.l = torch.nn.Linear(c_in, c_out)

        def forward(self, x):
            out = self.l(x)
            print(f"Output dtype: {out.dtype}")
            return out

    class LinearStack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stack = torch.nn.ModuleList()
            for _ in range(10):
                self.stack.append(Linear(10, 10))

        def forward(self, x):
            for b in self.stack:
                x = checkpoint(b, x)

            return x

    model = LinearStack().cuda()
    x = torch.rand(10, 10, requires_grad=True).cuda()

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        y = model(x)
        print("End of forward pass...")
    
    loss = torch.mean(y)
    loss.backward()

if __name__ == '__main__':
    main()

{'enabled': False, 'dtype': torch.bfloat16, 'cache_enabled': True}