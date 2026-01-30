import torch.nn as nn

import torch

inp_args = [
    torch.nn.Parameter(torch.randn(size, dtype=torch.float32), requires_grad=True)
    for size in [[1], [33, 1, 1, 1, 1], [33, 1, 1, 1, 1]]
]


def fn():
    v10_0 = torch.nn.Parameter(torch.empty([1], dtype=torch.float32), requires_grad=True)
    v8_0 = torch.nn.Parameter(torch.empty([33, 1, 1, 1, 1], dtype=torch.float32), requires_grad=True)
    v7_0 = torch.nn.Parameter(torch.empty([33, 1, 1, 1, 1], dtype=torch.float32), requires_grad=True)

    getitem = inp_args[0]
    getitem_1 = inp_args[1]
    getitem_2 = inp_args[2]

    matmul = torch.matmul(getitem, v10_0)
    cat = torch.cat((getitem_2, getitem_1, v7_0, v8_0), dim=3)
    mul = torch.mul(matmul, cat)
    linear_layer = torch.nn.Linear(in_features=1, out_features=36, bias=True)
    m9 = linear_layer(mul)
    tan = torch.tan(m9)

    return (tan,)


ret_eager = fn()
compiled = torch.compile(fn)
ret_compiled = compiled()

torch.testing.assert_close(ret_eager[0], ret_compiled[0])
# assert torch.allclose(ret_eager[0], ret_compiled[0]), '\n'.join(map(str, ["", ret_eager[0], ret_compiled[0]]))

# AssertionError: Tensor-likes are not close!
# 
# Mismatched elements: 4752 / 4752 (100.0%)
# Greatest absolute difference: nan at index (0, 0, 0, 0, 0) (up to 1e-05 allowed)
# Greatest relative difference: nan at index (0, 0, 0, 0, 0) (up to 1.3e-06 allowed)

# ...
# AssertionError:
# tensor([[[[[-9.6969e+00,  7.8080e-02, -9.2353e-01,  ...,  1.9866e+00,
#              3.4518e-01,  4.8450e+00],
#            [-1.1124e+00, -8.5060e-01,  3.2113e+00,  ..., -3.0666e-01,
#             -1.7086e-01, -6.7271e+00],
#            [-3.0328e-01, -7.8792e-01,  6.5650e-01,  ..., -9.0377e-01,
#             -6.4501e-01,  5.9481e+00],
#            [ 1.0956e-01,  6.0032e-01,  5.1952e-01,  ...,  1.4848e-01,
#              1.4129e-01, -4.0779e-01]]]] ...], )
# tensor([[[[[nan, nan, nan,  ..., nan, nan, nan],
#            [nan, nan, nan,  ..., nan, nan, nan],
#            [nan, nan, nan,  ..., nan, nan, nan],
#            [nan, nan, nan,  ..., nan, nan, nan]]]] ...],)