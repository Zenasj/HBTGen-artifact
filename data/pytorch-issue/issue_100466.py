import torch.nn as nn

py
import torch

torch.manual_seed(420)

input_tensor = torch.rand(3, 3)

class Model(torch.nn.Module):
    def forward(self, x):
        x = x * torch.tensor(x >= 0, dtype=torch.float32)
        return x

func = Model().to('cpu')

print(torch.tensor(input_tensor >= 0, dtype=torch.float32))
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])

with torch.no_grad():
    func.train(False)
    res1 = func(input_tensor) # without jit
    print(res1)
# tensor([[0.8054, 0.1990, 0.9759],
#         [0.1028, 0.3475, 0.1554],
#         [0.8856, 0.6876, 0.2506]])

    jit_func = torch.compile(func)
    res2 = jit_func(input_tensor)
    print(res2)
# tensor([[   nan,    nan,    nan],
#         [   nan,    nan,    nan],
#         [   nan,    nan, 0.2506]])

py
import torch

torch.manual_seed(420)

input_tensor = torch.rand(3, 3)

class Model(torch.nn.Module):
    def forward(self, x):
        return (x >= 0).to(torch.float32)

func = Model().to('cpu')

with torch.no_grad():
    func.train(False)
    res1 = func(input_tensor) # without jit
    print(res1)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])

    jit_func = torch.compile(func)
    res2 = jit_func(input_tensor)
    print(res2)
# tensor([[nan, nan, nan],
#         [nan, nan, nan],
#         [nan, nan, 1.]])

kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/5b/c5bcubr6yrbvnx73gevjlm24khhax3e2tzjnnvb47oxio6qm462z.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(8L); i0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0));
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(0));
            auto tmp2 = to_float_mask(tmp0 >= tmp1);
            auto tmp3 = (tmp2);
            auto tmp4 = tmp0 * tmp3;
            tmp4.store(out_ptr0 + static_cast<long>(i0));
        }
        #pragma omp simd simdlen(4)
        for(long i0=static_cast<long>(8L); i0<static_cast<long>(9L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(i0)];
            auto tmp1 = static_cast<float>(0);
            auto tmp2 = tmp0 >= tmp1;
            auto tmp3 = static_cast<float>(tmp2);
            auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
            out_ptr0[static_cast<long>(i0)] = tmp4;
        }
    }
}
''')

where