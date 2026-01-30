import torch.nn as nn

import torch
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms


class MyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        return x


class TensorParallelSize1Test(DTensorTestBase):

    @property
    def world_size(self) -> int:
        return 1

    @with_comms
    def test_auto_tp_plan(self):
        mesh = self.build_device_mesh()
        plan = {
            'linear': ColwiseParallel(),
        }
        module = MyModule()
        module = parallelize_module(module, mesh, plan)
        module = module.to(torch.cuda.current_device())
        x = torch.randn(10, 10)
        x = x.cuda()
        out = module(x)
        print(f'{out = }')


if __name__ == '__main__':
    run_tests()