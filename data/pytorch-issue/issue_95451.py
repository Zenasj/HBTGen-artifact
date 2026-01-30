import torch.nn as nn

import torch
import sys

i = sys.maxsize + 1
func_cls=torch.quantized_max_pool2d
# torch.nn.functional.max_pool1d
# torch.nn.functional.max_pool2d
# torch.nn.functional.max_pool3d
# torch.quantized_max_pool1d
# torch.quantized_max_pool2d


input = torch.full((1, 32, 32,), 0.5)
def test():
	func_cls(input, kernel_size=[i] , stride=[i], padding=0, dilation=[i], ceil_mode=True)
test()