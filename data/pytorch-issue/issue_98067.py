import torch

kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/hw/chwr6vy6e6sd25sfh42qtywkuf2emodexm2aomp3lbrcxwznfwyi.h"
extern "C" void kernel(long* out_ptr0)
{
   {
       auto tmp0 = static_cast<long>(8);
       out_ptr0[0] = tmp0;
   }
}
''')

async_compile.wait(globals())
del async_compile

def call(args):
   arg0_1, = args
   args.clear()
   kernel_cpp_0(c_void_p(arg0_1.data_ptr()))
   del arg0_1
   return (buf0, )

def benchmark_compiled_module():
   from torch._dynamo.testing import rand_strided
   from torch._inductor.utils import print_performance
   arg0_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
   print_performance(lambda: call([arg0_1]))

kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/hw/chwr6vy6e6sd25sfh42qtywkuf2emodexm2aomp3lbrcxwznfwyi.h"
extern "C" void kernel(long* out_ptr0,
                       long* out_ptr1)
{
    {
        auto tmp0 = static_cast<long>(8);
        out_ptr0[static_cast<long>(0)] = tmp0;
        out_ptr1[static_cast<long>(0)] = tmp0;
    }
}
''')

async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    buf0 = empty_strided((), (), device='cpu', dtype=torch.int64)
    kernel_cpp_0(c_void_p(buf0.data_ptr()), c_void_p(arg0_1.data_ptr()))
    del arg0_1
    return (buf0, )

def benchmark_compiled_module():
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    print_performance(lambda: call([arg0_1]))