import torch.nn as nn

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
torch._dynamo.config.debug_dir_root = '.'
torch._inductor.config.cpp.inject_relu_bug_TESTING_ONLY = 'compile_error'






from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_
        sigmoid = torch.sigmoid(l_x_);  l_x_ = None
        ones = torch.ones(2)
        mul = torch.mul(sigmoid, ones);  sigmoid = ones = None
        relu = torch.relu(mul);  mul = None
        zeros = torch.zeros(2)
        add = torch.add(relu, zeros);  relu = zeros = None
        round_1 = torch.ops.aten.round(add);  add = None
        return (round_1,)


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('b01abf7d0dfaf0ca12f8d2260c98f8c9b9d7bff9', 8)
    reader.tensor(buf0, (2,))  # L_x_
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='minify',
        save_dir='./run_2023_05_15_20_08_59_414896-pid_728954/minifier/checkpoints', autocast=False, backend='inductor')

from math import inf
import torch
from torch import tensor, device
import torch.fx as fxindent
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
from torch.nn import *
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config



class GeneratedReproTestCase(torch._dynamo.test_case.TestCase):
    @torch._dynamo.config.patch('debug_dir_root', '.')
    @torch._inductor.config.patch('cpp.inject_relu_bug_TESTING_ONLY', 'compile_error')
    def test_generated_repro_case(self):


        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()



            def forward(self, L_L_x_ : torch.Tensor):
                l_l_x_ = L_L_x_
                sigmoid = torch.sigmoid(l_l_x_);  l_l_x_ = None
                ones = torch.ones(2)
                mul = torch.mul(sigmoid, ones);  sigmoid = ones = None
                relu = torch.relu(mul);  mul = None
                zeros = torch.zeros(2)
                add = torch.add(relu, zeros);  relu = zeros = None
                round_1 = torch.ops.aten.round(add);  add = None
                return (round_1,)



        buf0 = torch.randn([2], dtype=torch.float32, device='cpu')  # L_L_x_ 

        # TODO(voz): Add dynamic, nopython, and all sorts of other configs
        optimized_result = torch._dynamo.optimize('inductor')(Repro())(buf0)
        eager_result = Repro()(buf0)
        self.assertEqual(optimized_result, eager_result)


if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests

    run_tests()