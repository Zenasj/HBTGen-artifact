import torch.nn as nn

import torch
from torch.utils.benchmark import Timer
from timeit import default_timer
input = torch.ones(10, 256, 128, 128)
w = torch.ones(256, 256, 3, 3)
b = torch.ones(256)
t=Timer(stmt="torch.nn.functional.conv2d(input, w, b)", language="python", timer=default_timer, globals={"input": input, "w": w, "b": b})
print(torch.__version__, t.blocked_autorange())