import torch.nn.functional as F
import torch
import torch.nn as nn

class JasperBlock(nn.Module):
    def __init__(self):
        super(JasperBlock, self).__init__()
        self.bn = nn.BatchNorm1d(3, eps=1e-3, momentum=0.1)
        
    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out)        
        return out

if __name__ == "__main__":

    model = JasperBlock()
    model.cuda()
    model.eval()
    model = model.half()

    x = torch.randn((3,3))*10
    x = x.cuda().half()
    out = model(x)
    model_jit = torch.jit.trace(model, (x,))
    out_jit = model_jit(x)

torch._C._jit_override_can_fuse_on_gpu(False)

import torch
import torch._C._te as te

input_str = """
graph(%x : Half(3, 3, strides=[3, 1], requires_grad=0, device=cpu),
       %weight : Half(3, strides=[1], requires_grad=0, device=cpu),
       %bias : Half(3, strides=[1], requires_grad=0, device=cpu),
       %running_mean : Half(3, strides=[1], requires_grad=0, device=cpu),
       %running_var : Half(3, strides=[1], requires_grad=0, device=cpu)):
   %5 : bool = prim::Constant[value=1]()
   %6 : float = prim::Constant[value=0.001]()
   %7 : float = prim::Constant[value=0.10000000000000001]()
   %8 : bool = prim::Constant[value=0]()
   %input.1 : Half(3, 3, strides=[3, 1], requires_grad=0, device=cpu) = aten::batch_norm(%x, %weight, %bias, %running_mean, %running_var, %8, %7, %6, %5)
   %10 : Half(3, 3, strides=[3, 1], requires_grad=0, device=cpu) = aten::relu(%input.1)
   return (%10)
"""


class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = torch._C._te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

with kernel_arena_scope():
    graph = torch._C.parse_ir(input_str)
    print(graph)
    kernel = te.TensorExprKernel(graph) # Fails
    print(kernel.get_code_text("asm"))

if os.getenv('MY_VAR_TO_TURN_OFF_GPU_FUSION'):
  torch._C._jit_override_can_fuse_on_gpu(False)