import torch
a = torch.tensor([[4.]])
b = torch.tensor([5.])
def forward(a, b):
    a = a.max(0).values
    c = torch.cat((a, b))
    c = c.round()
    b >= a[0]
    return c
print(forward(a, b))
fn_compiled = torch.compile(forward)
print(fn_compiled(a, b))

tensor([4., 5.])
tensor([-5.6321e+08,  5.0000e+00])

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf2 = empty_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    buf0 = as_strided(buf2, (1, ), (1, ))  # alias
    buf1 = as_strided(buf2, (1, ), (1, ), 1)  # alias
    buf3 = buf2; del buf2  # reuse
    kernel_cpp_0(c_void_p(buf3.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg1_1
    del buf0
    del buf1
    return (buf3, )

import torch
some_const = torch.tensor(6324, device='cuda:0')
def forward():
    a = torch.tensor([[0.6324]], device='cuda:0')
    ret = torch.cat((a, a), dim=0)
    some_const >= a[0] # Comment this line and the problem is gone.
    return  ret
print(forward())
fn_compiled = torch.compile(forward)
print(fn_compiled())