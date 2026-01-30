import torch.nn as nn
import random

with torch._dynamo.compiled_autograd.disable():
    torch.autograd.backward(...)

import os
import sys
import numpy as np

os.environ["TORCH_LOGS"] = "dynamo,graph_breaks,recompiles,graph_code,aot_joint_graph,aot_graphs"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
import torch
import torch._dynamo

os.environ["PT_HPU_LAZY_MODE"] = "0"
os.environ["ENABLE_CONSOLE"] = "1"
os.environ["LOG_LEVEL_ALL"] = "6"
os.environ["LOG_LEVEL_PT_EAGER"] = "1"

class MyModule(torch.nn.Module):
    def __init__(self, device):
        super(MyModule, self).__init__()
        self.layer = MyModule2(device)

    def forward(self, x):
        torch._dynamo.graph_break()
        res = x.cos() - x.sin()
        return self.layer(res)

class MyModule2(torch.nn.Module):
    def __init__(self, device):
        super(MyModule2, self).__init__()
        self.linear = torch.nn.Linear(2, 2, device=device)

    def forward(self, x):
        return self.linear(x).cos()


# stock pytorch checkpoint
def stock(args):
    device = args[0] if args else "cpu"

    from torch.utils.checkpoint import checkpoint
    
    if device == "cpu":
        def enable_compiled_autograd():
            def compiler_fn(gm):
                #return torch.compile(gm, backend='inductor', fullgraph=False)
                return torch.compile(gm, backend='eager', fullgraph=False)

            import functools
            import torch
            from torch._dynamo import compiled_autograd
            torch._C._dynamo.compiled_autograd.set_autograd_compiler(
                    functools.partial(compiled_autograd.AutogradCompilerInstance, compiler_fn)
                )

            torch._dynamo.reset()
            torch._dynamo.config.optimize_ddp = "python_reducer"
    else:
        import habana_frameworks.torch.dynamo.compile_backend
        from habana_frameworks.torch.dynamo.compile_backend.experimental import enable_compiled_autograd

    enable_compiled_autograd()

    m = MyModule(device)
    def fn(x):
        x = checkpoint(m.__call__, x)
        out = x + 2 
        out = out * 2 
        return out


    # fn = torch.compile(fn, backend="inductor" if device == "cpu" else "hpu_backend")
    fn = torch.compile(fn, backend="hpu_backend" if device == "hpu" else "inductor")
    for i in range(2):
        print(f"run {i} ..., ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
        x = torch.ones(2, 2, device=device, requires_grad=True)
        out = fn(x)
        print(out.cpu())
        print(f"backward {i} ..., bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        out.sum().backward()


if __name__ == "__main__":

    seed = 1000
    torch.manual_seed(seed)
    # habana_frameworks.torch.hpu.random.manual_seed(seed)

    if len(sys.argv) > 1:
        try:
            eval(f"{sys.argv[1]}({sys.argv[2:]})")
        except Exception as e:
            print(e)
            l = []
            for key, value in dict(locals()).items():
                if callable(value) and value.__module__ == __name__:
                    l.append(key)
            print(f"you can run one of the following functions: {l}")
        exit(0)

with torch._dynamo.compiled_autograd.disable():
    torch.autograd.backward(...)