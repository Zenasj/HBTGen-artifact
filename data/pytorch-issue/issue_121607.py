import torch

def func(a):
    x = torch.add(a, 1)
    return torch.reshape(torch.permute(x, (0, 2, 3, 1)), (2, 9216, 320))


a = torch.randn(2, 320, 96, 96).to(memory_format=torch.channels_last)
compiled_func = torch.compile(func)
with torch.no_grad():
    for _ in range(5):
        compiled_func(a)

cpp_fused_add_view_0 = async_compile.cpp_pybinding(['const float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_jiayisun/5t/c5tglt7hr54i6r4vlvkwzvrgdnsb3iaothtfcka6muxqjjb3rbxr.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(240)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(16L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9216L); x2+=static_cast<long>(16L))
                    {
                        float tmp0[16*16] __attribute__ ((aligned (16)));
                        at::vec::transpose_mxn<float,16,16>(in_ptr0 + static_cast<long>(x1 + (320L*x2) + (2949120L*x0)), static_cast<long>(320L), tmp0, 16);
                        for (long x1_inner = 0; x1_inner < 16; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(16L*x1_inner));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 + tmp3;
                            tmp4.store(out_ptr0 + static_cast<long>(x2 + (9216L*x1) + (9216L*x1_inner) + (2949120L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9216L); x1+=static_cast<long>(16L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(16L))
                    {
                        float tmp0[16*16] __attribute__ ((aligned (16)));
                        at::vec::transpose_mxn<float,16,16>(out_ptr0 + static_cast<long>(x1 + (9216L*x2) + (2949120L*x0)), static_cast<long>(9216L), tmp0, 16);
                        for (long x1_inner = 0; x1_inner < 16; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(16L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (320L*x1) + (320L*x1_inner) + (2949120L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2, 320, 96, 96), (2949120, 1, 30720, 320))
    buf0 = empty_strided_cpu((2, 320, 96, 96), (2949120, 9216, 96, 1), torch.float32)
    buf1 = empty_strided_cpu((2, 9216, 320), (2949120, 320, 1), torch.float32)
    cpp_fused_add_view_0(arg0_1, buf0, buf1)
    del arg0_1
    return (buf1, )