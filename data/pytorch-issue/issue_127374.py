import torch.nn as nn

from math import inf
import torch
from torch import tensor, device

from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config









from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, L_uint8_data_ : torch.Tensor):
        l_uint8_data_ = L_uint8_data_
        uint8_data = l_uint8_data_.to(torch.uint8);  l_uint8_data_ = None
        rshift = uint8_data >> 6
        first_elements = rshift & 3;  rshift = None
        rshift_1 = uint8_data >> 4
        second_elements = rshift_1 & 3;  rshift_1 = None
        rshift_2 = uint8_data >> 2
        third_elements = rshift_2 & 3;  rshift_2 = None
        fourth_elements = uint8_data & 3;  uint8_data = None
        stack = torch.stack((first_elements, second_elements, third_elements, fourth_elements), dim = -1);  first_elements = second_elements = third_elements = fourth_elements = None
        view = stack.view((16, 4));  stack = None
        return (view,)


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('bf1ec1fa8a065af5aa64669f811f7741ae598612', 16, device=device(type='cuda', index=0), dtype_hint=torch.uint8)
    reader.tensor(buf0, (16, 1), dtype=torch.uint8, is_leaf=True)  # L_uint8_data_
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='minify',
        save_dir='/home/james/bitnet/torch_compile_debug/run_2024_05_29_04_26_13_646526-pid_6449/minifier/checkpoints', autocast=False, backend='inductor')