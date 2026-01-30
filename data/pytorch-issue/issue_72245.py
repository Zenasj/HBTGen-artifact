3
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

class TEST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3,3, 3)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        return self.bn(self.conv(x))


rand_input = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    export_model = TEST().eval().cpu()
    print(export_model(rand_input).shape, export_model(rand_input).dtype)

    traced_script_module = torch.jit.trace(export_model, rand_input.cpu()) ## both options cannot work
    #traced_script_module = torch.jit.script(export_model) ## both options cannot work
    optimized_model = optimize_for_mobile(traced_script_module, backend='metal')
    print(torch.jit.export_opnames(optimized_model))
    optimized_model._save_for_lite_interpreter("traced_test.ptl")