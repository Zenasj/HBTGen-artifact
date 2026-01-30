import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.tensor(1) # <-- Adding this line raises a 'RuntimeError: at::functionalization::impl::isFunctionalTensor(self_) INTERNAL ASSERT FAILED'
        return x
    
export_options = torch.onnx.ExportOptions(dynamic_shapes=True) # Even with `dynamic_shapes=False`, the error persists
torch_model = MyModel()
torch_input = torch.randn(1, 1, 32, 32)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input, export_options=export_options)

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
torch._dynamo.config.specialize_int = True
torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.L__self___conv1 = Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
        self.L__self___conv2 = Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        self.L__self___fc1 = Linear(in_features=400, out_features=120, bias=True)
        self.L__self___fc2 = Linear(in_features=120, out_features=84, bias=True)
        self.L__self___fc3 = Linear(in_features=84, out_features=10, bias=True)

    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_
        l__self___conv1 = self.L__self___conv1(l_x_);  l_x_ = None
        relu = torch.nn.functional.relu(l__self___conv1);  l__self___conv1 = None
        x = torch.nn.functional.max_pool2d(relu, (2, 2));  relu = None
        l__self___conv2 = self.L__self___conv2(x);  x = None
        relu_1 = torch.nn.functional.relu(l__self___conv2);  l__self___conv2 = None
        x_1 = torch.nn.functional.max_pool2d(relu_1, 2);  relu_1 = None
        x_2 = torch.flatten(x_1, 1);  x_1 = None
        l__self___fc1 = self.L__self___fc1(x_2);  x_2 = None
        x_3 = torch.nn.functional.relu(l__self___fc1);  l__self___fc1 = None
        l__self___fc2 = self.L__self___fc2(x_3);  x_3 = None
        x_4 = torch.nn.functional.relu(l__self___fc2);  l__self___fc2 = None
        x_5 = self.L__self___fc3(x_4);  x_4 = None
        x_6 = torch.tensor(1)
        return (x_6,)

mod = Repro()

def load_args(reader):
    buf0 = reader.storage('3e9fbf7f2175dbf815b03cdb9328e36eaa8f0103', 4096)
    reader.tensor(buf0, (1, 1, 32, 32), is_leaf=True)  # L_x_
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='minify',
        save_dir='/home/a/cedie/surya_onnx/torch_compile_debug/run_2024_07_30_13_57_18_324762-pid_92116/minifier/checkpoints', autocast=False, backend=None)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.tensor(1)
        return x

torch_model = MyModel()
torch_input = torch.randn(1, 1, 32, 32)
onnx_program = torch.onnx.export(torch_model, (torch_input,), dynamo=True)
print(onnx_program)