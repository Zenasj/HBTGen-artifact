import torch

kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_chunyuan/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    {
        for(long i0=0; i0<1; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = decltype(tmp0)(2) / (decltype(tmp0)(1) + (decltype(tmp0)(-2) * tmp0).exp()) - decltype(tmp0)(1);
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=16; i0<16; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = std::tanh(tmp0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_1 = async_compile.cpp('''
#include "/tmp/torchinductor_chunyuan/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    {
        for(long i0=0; i0<1; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = decltype(tmp0)(2) / (decltype(tmp0)(1) + (decltype(tmp0)(-2) * tmp0).exp()) - decltype(tmp0)(1);
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=16; i0<16; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = std::tanh(tmp0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_2 = async_compile.cpp('''
#include "/tmp/torchinductor_chunyuan/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    {
        for(long i0=0; i0<1; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = decltype(tmp0)(2) / (decltype(tmp0)(1) + (decltype(tmp0)(-2) * tmp0).exp()) - decltype(tmp0)(1);
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=16; i0<16; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = std::tanh(tmp0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_3 = async_compile.cpp('''
#include "/tmp/torchinductor_chunyuan/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    {
        for(long i0=0; i0<1; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = decltype(tmp0)(2) / (decltype(tmp0)(1) + (decltype(tmp0)(-2) * tmp0).exp()) - decltype(tmp0)(1);
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=16; i0<16; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = std::tanh(tmp0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1 = args
    args.clear()
    buf0 = torch.ops.mkl._mkl_linear(arg15_1, arg2_1, arg0_1, arg1_1, 1)
    del arg0_1
    del arg15_1 
    del arg1_1
    del arg2_1
    buf1 = empty_strided((1, 16), (16, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del buf0
    buf2 = torch.ops.mkl._mkl_linear(buf1, arg5_1, arg3_1, arg4_1, 1)
    del arg3_1
    del arg4_1
    del arg5_1
    buf3 = buf1; del buf1  # reuse
    kernel_cpp_1(c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()))
    del buf2
    buf4 = torch.ops.mkl._mkl_linear(buf3, arg8_1, arg6_1, arg7_1, 1)
    del arg6_1
    del arg7_1
    del arg8_1
    buf5 = buf3; del buf3  # reuse
    kernel_cpp_2(c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    del buf4
    buf6 = torch.ops.mkl._mkl_linear(buf5, arg11_1, arg9_1, arg10_1, 1)
    del arg10_1
    del arg11_1
    del arg9_1
    buf7 = buf5; del buf5  # reuse
    kernel_cpp_3(c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del buf6
    buf8 = torch.ops.mkl._mkl_linear(buf7, arg14_1, arg12_1, arg13_1, 1)
    del arg12_1
    del arg13_1
    del arg14_1
    return (buf8, )

kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_chunyuan/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    {
        for(long i0=0; i0<1; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp1 = decltype(tmp0)(2) / (decltype(tmp0)(1) + (decltype(tmp0)(-2) * tmp0).exp()) - decltype(tmp0)(1);
            tmp1.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=16; i0<16; i0+=1)
        {
            auto tmp0 = in_out_ptr0[i0];
            auto tmp1 = std::tanh(tmp0);
            in_out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_1 = async_compile.cpp('''
#include "/tmp/torchinductor_chunyuan/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    {
        for(long i0=0; i0<1; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp1 = decltype(tmp0)(2) / (decltype(tmp0)(1) + (decltype(tmp0)(-2) * tmp0).exp()) - decltype(tmp0)(1);
            tmp1.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=16; i0<16; i0+=1)
        {
            auto tmp0 = in_out_ptr0[i0];
            auto tmp1 = std::tanh(tmp0);
            in_out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_2 = async_compile.cpp('''
#include "/tmp/torchinductor_chunyuan/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    {
        for(long i0=0; i0<1; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp1 = decltype(tmp0)(2) / (decltype(tmp0)(1) + (decltype(tmp0)(-2) * tmp0).exp()) - decltype(tmp0)(1);
            tmp1.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=16; i0<16; i0+=1)
        {
            auto tmp0 = in_out_ptr0[i0];
            auto tmp1 = std::tanh(tmp0);
            in_out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_3 = async_compile.cpp('''
#include "/tmp/torchinductor_chunyuan/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    {
        for(long i0=0; i0<1; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
            auto tmp1 = decltype(tmp0)(2) / (decltype(tmp0)(1) + (decltype(tmp0)(-2) * tmp0).exp()) - decltype(tmp0)(1);
            tmp1.store(in_out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=16; i0<16; i0+=1)
        {
            auto tmp0 = in_out_ptr0[i0];
            auto tmp1 = std::tanh(tmp0);
            in_out_ptr0[i0] = tmp1;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1 = args
    args.clear()
    buf0 = torch.ops.mkl._mkl_linear(arg15_1, arg2_1, arg0_1, arg1_1, 1)
    del arg0_1
    del arg15_1
    del arg1_1
    del arg2_1
    buf1 = as_strided(buf0, (1, 16), (16, 1)); del buf0  # reuse
    kernel_cpp_0(c_void_p(buf1.data_ptr()))
    buf2 = torch.ops.mkl._mkl_linear(buf1, arg5_1, arg3_1, arg4_1, 1)
    del arg3_1
    del arg4_1
    del arg5_1
    del buf1
    buf3 = as_strided(buf2, (1, 16), (16, 1)); del buf2  # reuse
    kernel_cpp_1(c_void_p(buf3.data_ptr()))
    buf4 = torch.ops.mkl._mkl_linear(buf3, arg8_1, arg6_1, arg7_1, 1)
    del arg6_1
    del arg7_1
    del arg8_1
    del buf3
    buf5 = as_strided(buf4, (1, 16), (16, 1)); del buf4  # reuse
    kernel_cpp_2(c_void_p(buf5.data_ptr()))
    buf6 = torch.ops.mkl._mkl_linear(buf5, arg11_1, arg9_1, arg10_1, 1)
    del arg10_1
    del arg11_1
    del arg9_1
    del buf5
    buf7 = as_strided(buf6, (1, 16), (16, 1)); del buf6  # reuse
    kernel_cpp_3(c_void_p(buf7.data_ptr()))
    buf8 = torch.ops.mkl._mkl_linear(buf7, arg14_1, arg12_1, arg13_1, 1)
    del arg12_1
    del arg13_1
    del arg14_1
    return (buf8, )