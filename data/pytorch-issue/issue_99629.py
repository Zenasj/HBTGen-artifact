py
import torch
import torch.nn as nn

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, dtype=torch.bfloat16)
        self.bn = nn.BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

input_tensor = torch.randn(1, 3, 64, 64, dtype=torch.bfloat16)

func = Model()

with torch.no_grad():
    func.train(False)
    res1 = func(input_tensor)
    # success
    jit_func = torch.compile(func)
    res2 = jit_func(input_tensor)
    # RuntimeError: ShapeProp error for: node=%l__self___conv : [#users=1] = call_module[target=L__self___conv](args = (%l_x_,), kwargs = {})

eager

aot_eager

inductor

bfloat16

nn.Conv2d

dtype