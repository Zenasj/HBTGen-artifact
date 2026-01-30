import torch.nn as nn

import torch
import torch._inductor.config as config

config.aot_inductor.debug_intermediate_value_printer = "2"
config.aot_inductor.filtered_kernel_names = "triton_poi_fused__to_copy_add_0"


class Model(torch.nn.Module):
    def forward(self, x):
        x = x.to(torch.float)
        return x + 1

model = Model().cuda()
x = torch.randn(10).cuda().to(torch.float8_e4m3fn)
_ = torch.compile(model, fullgraph=True)(x)

print("done")