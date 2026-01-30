import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch.fx.experimental.proxy_tensor import make_fx

def f(arg0_1):
    slice_1 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0)
    slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 20, 40)
    add = torch.ops.aten.add.Tensor(slice_2, 1)
    slice_3 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0)
    slice_4 = torch.ops.aten.slice.Tensor(slice_3, 1, 20, 40)
    slice_5 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0)
    slice_scatter = torch.ops.aten.slice_scatter.default(slice_5, add, 1, 20, 40)
    slice_scatter_1 = torch.ops.aten.slice_scatter.default(arg0_1, slice_scatter, 0, 0)
    slice_6 = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0)
    slice_7 = torch.ops.aten.slice.Tensor(slice_6, 1, 20, 40)
    slice_8 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0)
    slice_9 = torch.ops.aten.slice.Tensor(slice_8, 1, 1, 10)
    slice_10 = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0)
    slice_11 = torch.ops.aten.slice.Tensor(slice_10, 1, 1, 10)
    add_1 = torch.ops.aten.add.Tensor(slice_11, 2)
    slice_12 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0)
    slice_13 = torch.ops.aten.slice.Tensor(slice_12, 1, 2, 11)
    slice_14 = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0)
    slice_15 = torch.ops.aten.slice.Tensor(slice_14, 1, 2, 11)
    slice_16 = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0)
    slice_scatter_2 = torch.ops.aten.slice_scatter.default(slice_16, add_1, 1, 2, 11)
    slice_scatter_3 = torch.ops.aten.slice_scatter.default(slice_scatter_1, slice_scatter_2, 0, 0)
    slice_17 = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 0)
    slice_18 = torch.ops.aten.slice.Tensor(slice_17, 1, 2, 11)
    copy_ = torch.ops.aten.copy_.default(arg0_1, slice_scatter_3)
    return ()

x_ref = torch.ones([1, 64], device='cpu')
x_test = torch.ones([1, 64], device='cpu')
x_test2 = torch.ones([1, 64], device='cpu')
x_test3 = torch.ones([1, 64], device='cpu')

fx_g = make_fx(f)(x_test2)
f_compiled = compile_fx_inner(fx_g, [x_test3])

f(x_ref)
f_compiled([x_test])

print(x_test)
print(torch.abs(x_test - x_ref))
# Prints False
print(torch.allclose(x_test, x_ref))

cpp_fused_0 = async_compile.cpp(
    """
#include "/tmp/torchinductor_bzheng/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            auto tmp33 = in_ptr0[static_cast<long>(x0)];
            auto tmp0 = c10::convert<long>(x0);
            auto tmp1 = static_cast<long>(2);
            auto tmp2 = tmp0 >= tmp1;
            auto tmp3 = static_cast<long>(11);
            auto tmp4 = tmp0 < tmp3;
            auto tmp5 = tmp2 & tmp4;
            auto tmp6 = [&]
            {
                auto tmp7 = c10::convert<long>((-1L) + x0);
                auto tmp8 = static_cast<long>(20);
                auto tmp9 = tmp7 >= tmp8;
                auto tmp10 = static_cast<long>(40);
                auto tmp11 = tmp7 < tmp10;
                auto tmp12 = tmp9 & tmp11;
                auto tmp13 = [&]
                {
                    auto tmp14 = in_ptr0[static_cast<long>((-1L) + x0)];
                    auto tmp15 = static_cast<float>(1.0);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    return tmp16;
                }
                ;
                auto tmp17 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                auto tmp18 = in_ptr0[static_cast<long>((-1L) + x0)];
                auto tmp19 = tmp12 ? tmp17 : tmp18;
                auto tmp20 = static_cast<float>(2.0);
                auto tmp21 = decltype(tmp19)(tmp19 + tmp20);
                return tmp21;
            }
            ;
            auto tmp22 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
            auto tmp23 = static_cast<long>(20);
            auto tmp24 = tmp0 >= tmp23;
            auto tmp25 = static_cast<long>(40);
            auto tmp26 = tmp0 < tmp25;
            auto tmp27 = tmp24 & tmp26;
            auto tmp28 = [&]
            {
                auto tmp29 = in_ptr0[static_cast<long>(x0)];
                auto tmp30 = static_cast<float>(1.0);
                auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                return tmp31;
            }
            ;
            auto tmp32 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
            auto tmp34 = tmp27 ? tmp32 : tmp33;
            auto tmp35 = tmp5 ? tmp22 : tmp34;
            out_ptr1[static_cast<long>(x0)] = tmp35;
        }
    }
}
"""
)

async_compile.wait(globals())
del async_compile

def call(args):
    (arg0_1_1,) = args
    args.clear()
    assert_size_stride(arg0_1_1, (1, 64), (64, 1))
    cpp_fused_0(c_void_p(arg0_1_1.data_ptr()), c_void_p(arg0_1_1.data_ptr()))
    del arg0_1_1
    return ()

import torch
 
def f(a):
    a[:, 20:40] = a[:, 20:40] + 1
    a[:, 2:900025] = a[:, 1:900024] + 2
 
with torch.no_grad(): 
    a = torch.rand((1, 1000000), device="cuda")
    b = a.clone()
    f(a)
    compiled_fn = torch.compile(f)
    compiled_fn(b)
    print(a, flush=True)
    print(torch.abs(a - b), flush=True)
    # Prints False
    print(torch.allclose(a, b), flush=True)