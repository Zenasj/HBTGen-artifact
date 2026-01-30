import torch

#line 42: pow_1 = torch.ops.aten.pow.Tensor_Scalar(arg0_1, 5);  arg0_1 = None
#line 43: sin = torch.ops.aten.sin.default(pow_1);  pow_1 = None
cpp_fused_pow_sin_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/mq/cmqzxwuyo7ryvun3egqos5jq5ak4fue7d2jbopbqs7pgpkhdpfh4.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(48L); i0+=static_cast<long>(16L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i0));
            auto tmp1 = tmp0 * tmp0;
            auto tmp2 = tmp1 * tmp1;
            auto tmp3 = tmp2 * tmp0;
            auto tmp4 = tmp3.sin();
            tmp4.store(out_ptr0 + static_cast<long>(i0));
        }
        #pragma omp simd simdlen(8)
        for(long i0=static_cast<long>(48L); i0<static_cast<long>(60L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(i0)];
            auto tmp1 = decltype(tmp0)(tmp0 * tmp0);
            auto tmp2 = decltype(tmp1)(tmp1 * tmp1);
            auto tmp3 = decltype(tmp2)(tmp2 * tmp0);
            auto tmp4 = std::sin(tmp3);
            out_ptr0[static_cast<long>(i0)] = tmp4;
        }
    }
}
''')