cpp_fused_new_zeros_0 = async_compile.cpp('''
#include "/tmp/torchinductor_leslie/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(16L*(at::native::div_floor_integer(i1, 16L))); i0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>(static_cast<float>(0.0));
            tmp0.store(out_ptr0 + static_cast<long>(i0));
        }
        #pragma omp simd simdlen(8) 
        for(long i0=static_cast<long>(16L*(at::native::div_floor_integer(i1, 16L))); i0<static_cast<long>(i1); i0+=static_cast<long>(1L))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr0[static_cast<long>(i0)] = tmp0;
        }
    }
}
''')