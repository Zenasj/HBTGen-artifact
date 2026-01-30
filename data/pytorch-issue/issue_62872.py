import torch
import torch.nn as nn
import torch.utils.mobile_optimizer as mobile_optimizer
import torch.nn.functional as F

class Demo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        return x

model = Demo()
model = torch.quantization.convert(model)
model = torch.jit.script(model)
model = mobile_optimizer.optimize_for_mobile(model, backend='Metal')
model._save_for_lite_interpreter('model.ptl')

x = torch.rand((1, 3, 256, 256))
out = model(x)
print(out.shape)

3
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile


class TEST(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return  torch.pow(x, 2)


rand_input = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    export_model = TEST()
    export_model = export_model.eval()
    export_model = export_model.cpu()
    print(export_model(rand_input).shape)
    print(export_model(rand_input).dtype)

    #traced_script_module = torch.jit.trace(export_model, rand_input.cpu())
    traced_script_module = torch.jit.script(export_model)
    optimized_model = optimize_for_mobile(traced_script_module, backend='metal')
    print(torch.jit.export_opnames(optimized_model))
    optimized_model._save_for_lite_interpreter("traced_test.ptl")