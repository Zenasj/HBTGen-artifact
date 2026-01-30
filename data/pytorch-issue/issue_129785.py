# bug report -- with cuda (but not CPU), this crashes at large batch sizes
#
# torch version:  2.4.0a0+f70bd71a48.nv24.06
# cuda version :  12.5
#
# works at 1024, crashes during backwards() at 2048
# Also crashes at 2000; maybe due to very large activation temporaries crossing 2GB or 4GB?
b = 2048

import torch
import torch.nn as nn
import torch.nn.functional as F

print('torch version: ', torch.__version__)
print('cuda version : ', torch.version.cuda)

torch.set_default_device('cuda')

input_channels = 12
model = nn.Sequential(
    nn.Conv2d(12, 64, kernel_size=4, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2),
    nn.Conv2d(64, 128, kernel_size=4, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2),
    nn.Conv2d(128, 256, kernel_size=4, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2),
    nn.Conv2d(256, 512, kernel_size=4, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2),
    nn.Conv2d(512, 1024, kernel_size=4, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2),
    nn.Conv2d(1024, 2048, kernel_size=4, padding=1, bias=False),
    nn.ReLU(),
    nn.Flatten(start_dim=1),
    nn.Linear(24576, 120, bias=False)
)
print(model)
life_logits = model(torch.zeros(b, input_channels, 210, 160)).view(b, 3, 40)
train_loss = F.cross_entropy(life_logits[:,0], torch.zeros(b, dtype=torch.int64))

# crashes in here with:
# RuntimeError: CUDA error: an illegal memory access was encountered
train_loss.backward()

print(train_loss)

b = 2048

import torch
import torch.nn as nn
import torch.nn.functional as F

print('torch version: ', torch.__version__)
print('cuda version : ', torch.version.cuda)

torch.set_default_device('cuda')

from torch.utils._python_dispatch import TorchDispatchMode

class PrintMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs):
        print("func: ", func)
        return func(*args, **kwargs)

input_channels = 12
model = nn.Sequential(
    nn.Conv2d(12, 64, kernel_size=4, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2),
    # nn.Conv2d(64, 128, kernel_size=4, padding=1, bias=False),
    # nn.ReLU(),
    # nn.AvgPool2d(2),
    # nn.Conv2d(128, 256, kernel_size=4, padding=1, bias=False),
    # nn.ReLU(),
    # nn.AvgPool2d(2),
    # nn.Conv2d(256, 512, kernel_size=4, padding=1, bias=False),
    # nn.ReLU(),
    # nn.AvgPool2d(2),
    # nn.Conv2d(512, 1024, kernel_size=4, padding=1, bias=False),
    # nn.ReLU(),
    # nn.AvgPool2d(2),
    # nn.Conv2d(1024, 2048, kernel_size=4, padding=1, bias=False),
    # nn.ReLU(),
    # nn.Flatten(start_dim=1),
    # nn.Linear(24576, 120, bias=False)
)
print(model)
with PrintMode():
    life_logits = model(torch.zeros(b, input_channels, 210, 160))
    life_logits.sum().backward()

import torch
import torch.nn as nn
import torch.nn.functional as F

print('torch version: ', torch.__version__)
print('cuda version : ', torch.version.cuda)

torch.set_default_device('cuda')

inp = [torch.Size([2048, 64, 104, 79]), torch.Size([2048, 64, 209, 159]), [2, 2], [2, 2], [0, 0], False, True, None]
inp = [torch.randn(x, device='cuda') if isinstance(x, torch.Size) else x for x in inp]
torch.ops.aten.avg_pool2d_backward.default(*inp)