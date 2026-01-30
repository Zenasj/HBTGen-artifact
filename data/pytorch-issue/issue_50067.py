import torch
import timeit
from torch.utils.benchmark import Timer

timer = Timer("x.view({100, 5, 20});", setup="torch::Tensor x = torch::ones({10, 10, 100});", language="c++", timer=timeit.default_timer)
res = timer.collect_callgrind(number=10)