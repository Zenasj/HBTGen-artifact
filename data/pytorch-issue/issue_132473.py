cpp_fused_max_0 = async_compile.cpp_pybinding(['const bool*', 'bool*'], '''
#include "/tmp/torchinductor_root/xf/cxf75ftbahznonqovnsugw7v6sldrabizgtx3j4rhgdmu3r36wlu.h"
extern "C"  void kernel(const bool* in_ptr0,
                       bool* out_ptr0)
{
    {
        {
            bool tmp_acc0 = std::numeric_limits<bool>::min();
            at::vec::VecMask<float,1> tmp_acc0_vec = at::vec::VecMask<float,1>::from(std::numeric_limits<bool>::min());
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec = tmp_acc0_vec | tmp0;
            }
            #pragma omp simd simdlen(8) 
            for(long x0=static_cast<long>(112L); x0<static_cast<long>(125L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp_acc0_vec.all_zero());
            out_ptr0[static_cast<long>(0L)] = static_cast<bool>(tmp_acc0);
        }
    }
}
''')