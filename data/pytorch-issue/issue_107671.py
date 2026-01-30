import torch

def genf_int_float(x, y, use_transpose):
    if use_transpose:
        x, y = y, x
    x_int8 = torch.randint(-10, 10, (x, y), dtype=torch.int8, device='cuda')
    x_float = x_int8.to(torch.float32)
    if use_transpose:
        return x_int8.t(), x_float.t()
    return x_int8, x_float

def _test(m, k, n, transpose_a, transpose_b):
    a_int8, a_float = genf_int_float(m, k, transpose_a)
    b_int8, b_float = genf_int_float(k, n, transpose_b)
    c_int32 = torch._int_mm(a_int8, b_int8)
    c_float32 = torch.mm(a_float, b_float)
    print(c_int32[0])
    print(c_float32[0])
    torch.testing.assert_close(c_int32.float(), c_float32)

_test(17, 16, 16, False, True)