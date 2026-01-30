import torch

from torch.utils.benchmark import Timer

timer = Timer("x.view(-1);", setup="torch::Tensor x = torch::ones({1,2,3});", language="c++")
print(timer.blocked_autorange(min_run_time=5))
print(timer.collect_callgrind(number=10_000))