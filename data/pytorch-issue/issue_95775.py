import torch.nn as nn

# it seems that minifier failed to generate a minified code with `TORCHDYNAMO_REPRO_AFTER="aot"`
# so I will just put my MRE here.
import torch
import torch._dynamo
import torch._inductor.config
import logging

class ModuleTest2(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(torch.randn(1, 3, 224, 224))
    
    def forward(self, x):
        x = x + torch.ones_like(x)
        res = torch.nn.functional.conv2d(x, self.weight)
        res = res + torch.ones_like(res)
        return res

torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.output_code = True
torch._dynamo.config.verbose = True
torch._inductor.config.triton.convolution = "triton"
torch._inductor.config.debug = True

model = ModuleTest2().cuda()

model = torch.compile(model)
print(model)
print(model(torch.randn(1, 3, 224, 224).cuda()))