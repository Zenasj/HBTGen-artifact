import torch.nn as nn

import torch
import torch.nn.functional as F
class TestScriptsModule(torch.nn.Module):
    def __init__(self):
        super(TestScriptsModule,self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5)
        self.conv2 = torch.nn.Conv2d(20, 20, 5)

    def has_conv(self):
        if self.conv1 is None:
            return False
        if self.conv2 is None:
            return False
        return True

    def forward(self, img):
        ret = img
        if self.has_conv():
            ret = F.relu(self.conv1(img))
            ret = F.relu(self.conv2(ret))
        return ret

mm = TestScriptsModule()
inpudata = torch.rand((2,1,512,512))
ret = mm(inpudata)

mm2 = torch.jit.script(mm)
ret2 = mm2(inpudata)

mm.conv2 = None

mm3 = torch.jit.script(mm)
ret3 = mm3(inpudata)

print(ret2)

def forward(self, img):
        ret = img
        if self.conv1 is not None and self.conv2 is not None:
            ret = F.relu(self.conv1(img))
            ret = F.relu(self.conv2(ret))
        return ret