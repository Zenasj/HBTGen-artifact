import torch.nn as nn

from torch.utils._config_module import install_config_module
import sys

use_freezing = True
install_config_module(sys.modules[__name__])

class DictLikeClass:
    def __getitem__(self, key):
        return getattr(sys.modules[__name__], key)

    def __setitem__(self, key, value):
        return setattr(sys.modules[__name__], key, value)

configuration_flags = DictLikeClass()

import torch
from torch._functorch.aot_autograd import aot_module_simplified
import torch._dynamo as dynamo

import config as myconfig

def my_backend(gm, sample_inputs):
    def my_compiler(gm, sample_inputs):
        if tracing_context := torch._guards.TracingContext.try_get():
            fw_metadata = tracing_context.fw_metadata
            params_flat = tracing_context.params_flat
            print(f"my_compiler: params_flat= {params_flat}")
        return gm.forward

    print(f"myconfig freezing flag : {myconfig.use_freezing}")
    return aot_module_simplified(gm, sample_inputs, fw_compiler=my_compiler)

class MyClass(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p1 = torch.nn.Parameter(torch.randn([2, 3]))
        self.p2 = torch.nn.Parameter(torch.randn([2, 3]))

    #@torch._inductor.config.patch("freezing", True)
    @myconfig.patch("use_freezing", True)
    def forward(self, x):
        t = self.p1 + x
        out = t / self.p2
        return out
    
mod = MyClass()
compiled_mod = torch.compile(mod, backend=my_backend)
inp = torch.randn([2, 3])
with torch.no_grad():
    r = compiled_mod(inp)

def is_parameter_freezing():
    return torch._inductor.config.freezing and not torch.is_grad_enabled()