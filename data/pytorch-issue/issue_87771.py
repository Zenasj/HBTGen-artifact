import torch.nn as nn

import torch
import torchvision
import torch._dynamo
import torch.fx.experimental.optimization as optimization
import copy
import time
import torch.profiler as profiler
from torch.fx import symbolic_trace
from torch._inductor import config
config.debug = True

#torch.manual_seed(2020)

model = torchvision.models.resnet50().eval()

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torchvision.models.resnet50().conv1
        self.bn = torchvision.models.resnet50().bn1
        self.relu = torchvision.models.resnet50().relu
        self.pool = torchvision.models.resnet50().maxpool
        self.layer1 = torchvision.models.resnet50().layer1
        '''
        self.layer2 = torchvision.models.resnet50().layer2
        self.layer3 = torchvision.models.resnet50().layer3
        self.layer4 = torchvision.models.resnet50().layer4
        self.pool2 = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)
        '''

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.layer1[0](x)

        return x


#model =  torchvision.models.resnet50().eval()

model= Model().eval()

#model = model.to(memory_format=torch.channels_last).eval()

opt_model = torch._dynamo.optimize('inductor')(model)

#opt_model = model
#torch._C._jit_set_texpr_fuser_enabled(False)
batch_size = 20
x = torch.randn(batch_size, 3, 224, 224)

#x = x.to(memory_format=torch.channels_last)

warm_up = 200

with torch.no_grad():
    for i in range(warm_up):
        y1 = opt_model(x)
print("begin running...............")

num_iter = 200
fwd = 0
with torch.no_grad():
    t1  = time.time()
    for i in range(num_iter):
        y2 = opt_model(x)
    t2 = time.time()
    fwd = fwd + (t2 - t1)

print(torch.equal(y1, y2))
avg_time = fwd / num_iter * 1000
print("batch_size = %d, avg time is %0.3f (ms) fps:%f"%(batch_size, avg_time, batch_size  * num_iter / fwd))

from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


kernel0 = async_compile.cpp('''
#include "/tmp/torchinductor_xiaobing/i5/ci5zbqbzeij2usetynv7oczewshegubkvtpswwuumpp6xjync55y.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    #pragma omp parallel num_threads(40)
    {
        #pragma omp for
        for(long i0=0; i0<1280; ++i0)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<56; ++i1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<56; ++i2)
                {
                    {
                        {
                            auto tmp0 = static_cast<long>((-1) + (2*i1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(112);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = static_cast<long>((-1) + (2*i2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            float tmp11 = -std::numeric_limits<float>::infinity();
                            if(tmp10)
                            {
                                auto tmp12 = in_ptr0[(-113) + (2*i2) + (224*i1) + (12544*i0)];
                                tmp11 = tmp12;
                            }
                            auto tmp13 = static_cast<long>((-113) + (2*i2) + (224*i1));
                            auto tmp14 = static_cast<long>(2*i2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            float tmp19 = -std::numeric_limits<float>::infinity();
                            if(tmp18)
                            {
                                auto tmp20 = in_ptr0[(-112) + (2*i2) + (224*i1) + (12544*i0)];
                                tmp19 = tmp20;
                            }
                            auto tmp21 = static_cast<long>((-112) + (2*i2) + (224*i1));
                            auto tmp22 = tmp19 > tmp11;
                            auto tmp23 = tmp22 ? tmp21 : tmp13;
                            auto tmp24 = std::max(tmp19, tmp11);
                            auto tmp25 = static_cast<long>(1 + (2*i2));
                            auto tmp26 = tmp25 >= tmp1;
                            auto tmp27 = tmp25 < tmp3;
                            auto tmp28 = tmp26 & tmp27;
                            auto tmp29 = tmp5 & tmp28;
                            float tmp30 = -std::numeric_limits<float>::infinity();
                            if(tmp29)
                            {
                                auto tmp31 = in_ptr0[(-111) + (2*i2) + (224*i1) + (12544*i0)];
                                tmp30 = tmp31;
                            }
                            auto tmp32 = static_cast<long>((-111) + (2*i2) + (224*i1));
                            auto tmp33 = tmp30 > tmp24;
                            auto tmp34 = tmp33 ? tmp32 : tmp23;
                            auto tmp35 = std::max(tmp30, tmp24);
                            auto tmp36 = static_cast<long>(2*i1);
                            auto tmp37 = tmp36 >= tmp1;
                            auto tmp38 = tmp36 < tmp3;
                            auto tmp39 = tmp37 & tmp38;
                            auto tmp40 = tmp39 & tmp9;
                            float tmp41 = -std::numeric_limits<float>::infinity();
                            if(tmp40)
                            {
                                auto tmp42 = in_ptr0[(-1) + (2*i2) + (224*i1) + (12544*i0)];
                                tmp41 = tmp42;
                            }
                            auto tmp43 = static_cast<long>((-1) + (2*i2) + (224*i1));
                            auto tmp44 = tmp41 > tmp35;
                            auto tmp45 = tmp44 ? tmp43 : tmp34;
                            auto tmp46 = std::max(tmp41, tmp35);
                            auto tmp47 = tmp39 & tmp17;
                            float tmp48 = -std::numeric_limits<float>::infinity();
                            if(tmp47)
                            {
                                auto tmp49 = in_ptr0[(2*i2) + (224*i1) + (12544*i0)];
                                tmp48 = tmp49;
                            }
                            auto tmp50 = static_cast<long>((2*i2) + (224*i1));
                            auto tmp51 = tmp48 > tmp46;
                            auto tmp52 = tmp51 ? tmp50 : tmp45;
                            auto tmp53 = std::max(tmp48, tmp46);
                            auto tmp54 = tmp39 & tmp28;
                            float tmp55 = -std::numeric_limits<float>::infinity();
                            if(tmp54)
                            {
                                auto tmp56 = in_ptr0[1 + (2*i2) + (224*i1) + (12544*i0)];
                                tmp55 = tmp56;
                            }
                            auto tmp57 = static_cast<long>(1 + (2*i2) + (224*i1));
                            auto tmp58 = tmp55 > tmp53;
                            auto tmp59 = tmp58 ? tmp57 : tmp52;
                            auto tmp60 = std::max(tmp55, tmp53);
                            auto tmp61 = static_cast<long>(1 + (2*i1));
                            auto tmp62 = tmp61 >= tmp1;
                            auto tmp63 = tmp61 < tmp3;
                            auto tmp64 = tmp62 & tmp63;
                            auto tmp65 = tmp64 & tmp9;
                            float tmp66 = -std::numeric_limits<float>::infinity();
                            if(tmp65)
                            {
                                auto tmp67 = in_ptr0[111 + (2*i2) + (224*i1) + (12544*i0)];
                                tmp66 = tmp67;
                            }
                            auto tmp68 = static_cast<long>(111 + (2*i2) + (224*i1));
                            auto tmp69 = tmp66 > tmp60;
                            auto tmp70 = tmp69 ? tmp68 : tmp59;
                            auto tmp71 = std::max(tmp66, tmp60);
                            auto tmp72 = tmp64 & tmp17;
                            float tmp73 = -std::numeric_limits<float>::infinity();
                            if(tmp72)
                            {
                                auto tmp74 = in_ptr0[112 + (2*i2) + (224*i1) + (12544*i0)];
                                tmp73 = tmp74;
                            }
                            auto tmp75 = static_cast<long>(112 + (2*i2) + (224*i1));
                            auto tmp76 = tmp73 > tmp71;
                            auto tmp77 = tmp76 ? tmp75 : tmp70;
                            auto tmp78 = std::max(tmp73, tmp71);
                            auto tmp79 = tmp64 & tmp28;
                            float tmp80 = -std::numeric_limits<float>::infinity();
                            if(tmp79)
                            {
                                auto tmp81 = in_ptr0[113 + (2*i2) + (224*i1) + (12544*i0)];
                                tmp80 = tmp81;
                            }
                            auto tmp82 = static_cast<long>(113 + (2*i2) + (224*i1));
                            auto tmp83 = tmp80 > tmp78;
                            auto tmp84 = tmp83 ? tmp82 : tmp77;
                            auto tmp85 = std::max(tmp80, tmp78);
                            out_ptr0[i2 + (56*i1) + (3136*i0)] = tmp85;
                        }
                    }
                }
            }
        }
        #pragma omp for  collapse(2)
        for(long i0=0; i0<20; ++i0)
        {
            for(long i1=0; i1<64; ++i1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<3136; ++i2)
                {
                    {
                        {
                            auto tmp0 = out_ptr0[i2 + (3136*i1) + (200704*i0)];
                            out_ptr1[i1 + (64*i2) + (200704*i0)] = tmp0;
                            out_ptr2[i1 + (64*i2) + (200704*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel1 = async_compile.cpp('''
#include "/tmp/torchinductor_xiaobing/i5/ci5zbqbzeij2usetynv7oczewshegubkvtpswwuumpp6xjync55y.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(40)
    {
        #pragma omp for
        for(long i0=0; i0<62720; ++i0)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<64; ++i1)
            {
                {
                    {
                        auto tmp0 = in_ptr0[i1 + (64*i0)];
                        auto tmp1 = in_ptr1[i1];
                        auto tmp3 = in_ptr2[i1];
                        auto tmp11 = in_ptr3[i1];
                        auto tmp13 = in_ptr4[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = std::sqrt(tmp5);
                        auto tmp7 = 1 / tmp6;
                        auto tmp8 = static_cast<float>(1);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = tmp2 * tmp9;
                        auto tmp12 = tmp10 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp15 = tmp14 * (tmp14>0);
                        out_ptr0[i1 + (64*i0)] = tmp15;
                    }
                }
            }
        }
        #pragma omp for  collapse(2)
        for(long i0=0; i0<20; ++i0)
        {
            for(long i1=0; i1<64; ++i1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<3136; ++i2)
                {
                    {
                        {
                            auto tmp0 = out_ptr0[i1 + (64*i2) + (200704*i0)];
                            out_ptr1[i2 + (3136*i1) + (200704*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel2 = async_compile.cpp('''
#include "/tmp/torchinductor_xiaobing/i5/ci5zbqbzeij2usetynv7oczewshegubkvtpswwuumpp6xjync55y.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(40)
    {
        #pragma omp for  collapse(2)
        for(long i0=0; i0<20; ++i0)
        {
            for(long i1=0; i1<64; ++i1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<3136; ++i2)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i2 + (3136*i1) + (200704*i0)];
                            auto tmp1 = in_ptr1[i1];
                            auto tmp3 = in_ptr2[i1];
                            auto tmp11 = in_ptr3[i1];
                            auto tmp13 = in_ptr4[i1];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp6 = std::sqrt(tmp5);
                            auto tmp7 = 1 / tmp6;
                            auto tmp8 = static_cast<float>(1);
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp10 = tmp2 * tmp9;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp14 = tmp12 + tmp13;
                            auto tmp15 = tmp14 * (tmp14>0);
                            out_ptr1[i1 + (64*i2) + (200704*i0)] = tmp15;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel3 = async_compile.cpp('''
#include "/tmp/torchinductor_xiaobing/i5/ci5zbqbzeij2usetynv7oczewshegubkvtpswwuumpp6xjync55y.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       const float* __restrict__ in_ptr8,
                       const float* __restrict__ in_ptr9)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(40)
    {
        #pragma omp for
        for(long i0=0; i0<62720; ++i0)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<256; ++i1)
            {
                {
                    {
                        auto tmp0 = in_ptr0[i1 + (256*i0)];
                        auto tmp1 = in_ptr1[i1];
                        auto tmp3 = in_ptr2[i1];
                        auto tmp11 = in_ptr3[i1];
                        auto tmp13 = in_ptr4[i1];
                        auto tmp15 = in_ptr5[i1 + (256*i0)];
                        auto tmp16 = in_ptr6[i1];
                        auto tmp18 = in_ptr7[i1];
                        auto tmp24 = in_ptr8[i1];
                        auto tmp26 = in_ptr9[i1];
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = std::sqrt(tmp5);
                        auto tmp7 = 1 / tmp6;
                        auto tmp8 = static_cast<float>(1);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = tmp2 * tmp9;
                        auto tmp12 = tmp10 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp17 = tmp15 - tmp16;
                        auto tmp19 = tmp18 + tmp4;
                        auto tmp20 = std::sqrt(tmp19);
                        auto tmp21 = 1 / tmp20;
                        auto tmp22 = tmp21 * tmp8;
                        auto tmp23 = tmp17 * tmp22;
                        auto tmp25 = tmp23 * tmp24;
                        auto tmp27 = tmp25 + tmp26;
                        auto tmp28 = tmp14 + tmp27;
                        out_ptr0[i1 + (256*i0)] = tmp28;
                    }
                }
            }
        }
        #pragma omp for  collapse(2)
        for(long i0=0; i0<20; ++i0)
        {
            for(long i1=0; i1<256; ++i1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<3136; ++i2)
                {
                    {
                        {
                            auto tmp0 = out_ptr0[i1 + (256*i2) + (802816*i0)];
                            auto tmp1 = tmp0 * (tmp0>0);
                            out_ptr1[i1 + (256*i2) + (802816*i0)] = tmp1;
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1 = args
    args.clear()
    buf0 = aten.convolution(arg25_1, arg0_1, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf0, (20, 64, 112, 112), (802816, 12544, 112, 1))
    del arg0_1
    del arg25_1
    buf1 = empty_strided((20, 64, 56, 56), (200704, 3136, 56, 1), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((20, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((20, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    kernel0(c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf14.data_ptr()))
    del buf0
    del buf1
    buf3 = aten.convolution(buf11, arg1_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf3, (20, 64, 56, 56), (200704, 1, 3584, 64))
    del arg1_1
    buf4 = buf3; del buf3  # reuse
    buf12 = as_strided(buf11, (20, 64, 56, 56), (200704, 3136, 56, 1)); del buf11  # reuse
    kernel1(c_void_p(buf4.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf12.data_ptr()))
    del arg13_1
    del arg14_1
    del arg2_1
    del arg3_1
    del buf4
    buf5 = aten.convolution(buf12, arg4_1, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf5, (20, 64, 56, 56), (200704, 3136, 56, 1))
    del arg4_1
    buf6 = buf5; del buf5  # reuse
    buf13 = as_strided(buf12, (20, 64, 56, 56), (200704, 1, 3584, 64)); del buf12  # reuse
    kernel2(c_void_p(buf6.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg16_1
    del arg17_1
    del arg5_1
    del arg6_1
    del buf6
    buf7 = aten.convolution(buf13, arg7_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf7, (20, 256, 56, 56), (802816, 1, 14336, 256))
    del arg7_1
    del buf13
    buf8 = aten.convolution(buf14, arg10_1, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1)
    assert_size_stride(buf8, (20, 256, 56, 56), (802816, 1, 14336, 256))
    del arg10_1
    del buf14
    buf9 = buf7; del buf7  # reuse
    buf10 = buf9; del buf9  # reuse
    kernel3(c_void_p(buf10.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg12_1.data_ptr()))
    del arg11_1
    del arg12_1
    del arg19_1
    del arg20_1
    del arg22_1
    del arg23_1
    del arg8_1
    del arg9_1
    return (buf10, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg16_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg19_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg22_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg25_1 = rand_strided((20, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1]))