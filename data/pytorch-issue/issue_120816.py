import torch

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    assert_size_stride(arg2_1, (s0, s1), (s1, 1))
    assert_size_stride(arg3_1, (1, ), (1, ))
    u3 = arg3_1.item()
    buf0 = None
    del arg3_1
    buf3 = empty_strided_cpu((u0 + s0, s1), (s1, 1), torch.float32)
    buf1 = reinterpret_tensor(buf3, (((-1)*(min(0, u0 + s0))) + (min(s0, u0 + s0)), s1), (s1, 1), s1*(min(0, u0 + s0)))  # alias
    buf2 = reinterpret_tensor(buf3, (u0 + s0 + ((-1)*(min(s0, u0 + s0))), s1), (s1, 1), s1*(min(s0, u0 + s0)))  # alias
    cpp_fused_cat_new_ones_0(arg2_1, buf1, buf2, s0, s1, u0)
    del arg2_1
    return (buf3, )

def f(x, s0):
	u0 = x.item()
	torch._check(u0 == s0)
	torch._check(s0 == 3)

y = x.item()
z = y + 2
torch._check(z == 3)