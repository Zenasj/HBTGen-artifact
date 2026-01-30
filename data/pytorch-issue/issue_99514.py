import torch.nn as nn

convit_base

import torch
import torch._dynamo
import torch._dynamo.config as config

config.dynamic_shapes=True
torch._dynamo.config.assume_static_by_default=False
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        B, N, C = x.shape
        return self.get_rel_indices(N)

    def get_rel_indices(self, num_patches: int) -> torch.Tensor:
        img_size = int(num_patches ** .5)
        #rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size)
        return ind

model = Model().eval()
opt_model = torch._dynamo.optimize('inductor')(model)

x = torch.randn(8, 8, 8)
ref = model(x)
with torch.no_grad():
    for i in range(3):
        out = opt_model(x)

kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_xiaobing/x5/cx5442c6dcuxsrrlnqi476yzjlgc6g53ukppuaettiyp6dszhmr4.h"
extern "C" void kernel(long* out_ptr0,
                       const long ks0)
{
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(std::floor(std::sqrt(ks0))); i0+=static_cast<long>(1L))
        {
            auto tmp0 = static_cast<long>(i0);
            out_ptr0[static_cast<long>(i0)] = tmp0;
        }
    }
}
''')