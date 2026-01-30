cpp_fused_clone_4 = async_compile.cpp('''
#include <ATen/record_function.h>
#include "/tmp/torchinductor_root/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    RECORD_FUNCTION("graph_0_cpp_fused_clone_4", c10::ArrayRef<c10::IValue>({}));
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(16L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(331776L); i1+=static_cast<long>(16L))
            {
                float tmp0[16*16] __attribute__ ((aligned (16)));
                at::vec::transpose_mxn<float,16,16>(in_ptr0 + static_cast<long>(i0 + (16L*i1)), static_cast<long>(16L), tmp0, 16);
                for (long i0_inner = 0; i0_inner < 16; i0_inner++)
                {
                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(16L*i0_inner));
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(in_ptr1[static_cast<long>(i0 + i0_inner)]));
                    auto tmp3 = tmp1 + tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(i1 + (331776L*i0) + (331776L*i0_inner)));
                }
            }
        }
    }
}
''')

buf16 = buf14; del buf14  # reuse
cpp_fused_clone_4(c_void_p(buf15.data_ptr()), c_void_p(_frozen_param87.data_ptr()), c_void_p(buf16.data_ptr()))

cpp_fused_clone_4 = async_compile.cpp('''
#include <ATen/record_function.h>
#include "/tmp/torchinductor_liaoxuan/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                      const float* in_ptr1,
                      float* out_ptr0)
{
  RECORD_FUNCTION("graph_0_cpp_fused_clone_4", c10::ArrayRef<c10::IValue>({}));
   {
       for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L); i0+=static_cast<long>(16L))
       {
           #pragma GCC ivdep
           for(long i1=static_cast<long>(0L); i1<static_cast<long>(331776L); i1+=static_cast<long>(1L))
           {
               auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0 + (16L*i1)));
               auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(i0));
               auto tmp2 = tmp0 + tmp1;
               { __at_align__ float tmpbuf[16*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long i0_inner = 0; i0_inner < 16; i0_inner++) out_ptr0[static_cast<long>(i1 + (331776L*i0) + (331776L*i0_inner))] = tmpbuf[i0_inner]; }
           }
       }
   }
}
''')