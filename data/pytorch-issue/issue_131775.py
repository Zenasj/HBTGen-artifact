import torch
torch._dynamo.config.capture_scalar_outputs = True

@torch.compile()
def foo(x):
    return x.sum().item()

foo(torch.rand([20]))

cpp_fused_scalar_tensor_1 = async_compile.cpp_pybinding(['float*'], '''
#include "/tmp/torchinductor_yhao/ky/cky2bufythacofebk7ujv36e4pxyqcqbpsy5r4vojoprjiwcwfxf.h"
extern "C"  void kernel(float* out_ptr0)
{
    {
        auto tmp0 = zuf0;
        auto tmp1 = c10::convert<float>(tmp0);
        out_ptr0[static_cast<long>(0L)] = tmp1;
    }
}
''')
def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (20, ), (1, ))
    buf0 = empty_strided_cpu((), (), torch.float16)
    cpp_fused_sum_0(arg0_1, buf0)
    del arg0_1
    zuf0 = buf0.item()
    buf1 = None
    del buf0
    buf2 = empty_strided_cpu((), (), torch.float32)
    cpp_fused_scalar_tensor_1(buf2)
    return (buf2, )

cpp_fused_scalar_tensor_1 = async_compile.cpp_pybinding(['double*', 'const double'], '''
#include "/tmp/torchinductor_yhao/ky/cky2bufythacofebk7ujv36e4pxyqcqbpsy5r4vojoprjiwcwfxf.h"
extern "C"  void kernel(double* out_ptr0,
                       const double ks0)
{
    {
        auto tmp0 = ks0;
        auto tmp1 = c10::convert<double>(tmp0);
        out_ptr0[static_cast<long>(0L)] = tmp1;
    }
}
''')
def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (20, ), (1, ))
    buf0 = empty_strided_cpu((), (), torch.float16)
    cpp_fused_sum_0(arg0_1, buf0)
    del arg0_1
    zuf0 = buf0.item()
    buf1 = None
    del buf0
    buf2 = empty_strided_cpu((), (), torch.float64)
    cpp_fused_scalar_tensor_1(buf2, zuf0)
    return (buf2, )