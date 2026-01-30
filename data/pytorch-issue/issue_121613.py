import torch

def func(a, b):
    x1 = torch.add(a, 1)
    x2 = torch.add(b, 2)
    return torch.cat((x1, x2), dim=1)

a = torch.randn(2, 320, 96, 96).to(memory_format=torch.channels_last)
b = torch.randn(2, 320, 96, 96).to(memory_format=torch.channels_last)
compiled_func = torch.compile(func)
with torch.no_grad():
    for _ in range(5):
        compiled_func(a, b)

cpp_fused_add_cat_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*'], '''
#include "/tmp/torchinductor_jiayisun/5t/c5tglt7hr54i6r4vlvkwzvrgdnsb3iaothtfcka6muxqjjb3rbxr.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                            tmp4.store(out_ptr0 + static_cast<long>(x2 + (9216L*x1) + (9216L*x1_inner) + (5898240L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(16L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9216L); x2+=static_cast<long>(16L))
                    {
                        float tmp0[16*16] __attribute__ ((aligned (16)));
                        at::vec::transpose_mxn<float,16,16>(in_ptr1 + static_cast<long>(x1 + (320L*x2) + (2949120L*x0)), static_cast<long>(320L), tmp0, 16);
                        for (long x1_inner = 0; x1_inner < 16; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(16L*x1_inner));
                            auto tmp2 = static_cast<float>(2.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 + tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (9216L*x1) + (9216L*x1_inner) + (5898240L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(16L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9216L); x2+=static_cast<long>(16L))
                    {
                        float tmp1[16*16] __attribute__ ((aligned (16)));
                        for (long x1_inner = 0; x1_inner < 16; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (9216L*x1) + (9216L*x1_inner) + (5898240L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(16L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,16,16>(tmp1, 16, out_ptr2 + static_cast<long>(x1 + (640L*x2) + (5898240L*x0)), static_cast<long>(640L));
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
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 320, 96, 96), (2949120, 1, 30720, 320))
    assert_size_stride(arg1_1, (2, 320, 96, 96), (2949120, 1, 30720, 320))
    buf2 = empty_strided_cpu((2, 640, 96, 96), (5898240, 9216, 96, 1), torch.float32)
    buf0 = reinterpret_tensor(buf2, (2, 320, 96, 96), (5898240, 9216, 96, 1), 0)  # alias
    buf1 = reinterpret_tensor(buf2, (2, 320, 96, 96), (5898240, 9216, 96, 1), 2949120)  # alias
    buf3 = empty_strided_cpu((2, 640, 96, 96), (5898240, 1, 61440, 640), torch.float32)
    cpp_fused_add_cat_0(arg0_1, arg1_1, buf2, buf0, buf1, buf3)
    del arg0_1
    del arg1_1
    return (buf3, )