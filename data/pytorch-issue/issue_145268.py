import torch.nn as nn

import torch

inp_args = [
    torch.nn.Parameter(torch.randn([23, 1, 1, 1, 1], dtype=torch.float32), requires_grad=True)
]


def fn():
    getitem = inp_args[0]
    interpolate = torch.nn.functional.interpolate(getitem, size=[1, 1, 1], scale_factor=None, mode='trilinear',
                                                  align_corners=None, recompute_scale_factor=None, antialias=False)
    linear_layer = torch.nn.Linear(in_features=1, out_features=34, bias=True)
    m2 = linear_layer(getitem)
    interpolate_1 = torch.nn.functional.interpolate(m2, size=[1, 39, 34], scale_factor=None, mode='trilinear',
                                                    align_corners=None, recompute_scale_factor=None, antialias=False)
    mean = interpolate_1.mean(0)
    gt = torch.gt(m2, interpolate_1)
    return (interpolate, mean, gt)


ret_eager = fn()
compiled = torch.compile(fn)
ret_compiled = compiled()

torch.testing.assert_close(ret_eager[1], ret_compiled[1])
# assert torch.allclose(ret_eager[1], ret_compiled[1]), '\n'.join(map(str, ["", ret_eager[1], ret_compiled[1]]))
# torch.testing.assert_close(ret_eager[2], ret_compiled[2])
# assert torch.allclose(ret_eager[2], ret_compiled[2]), '\n'.join(map(str, ["", ret_eager[2], ret_compiled[2]]))

# AssertionError: Tensor-likes are not close!
# 
# Mismatched elements: 1326 / 1326 (100.0%)
# Greatest absolute difference: 1.6459496021270752 at index (0, 0, 0, 8) (up to 1e-05 allowed)
# Greatest relative difference: 12.975427627563477 at index (0, 0, 0, 33) (up to 1.3e-06 allowed)

# ...
# AssertionError:
# tensor([[[[-0.0624,  0.5362, -0.5600,  ...,  0.7437, -0.3323, -0.2239],
#           [-0.0624,  0.5362, -0.5600,  ...,  0.7437, -0.3323, -0.2239],
#           [-0.0624,  0.5362, -0.5600,  ...,  0.7437, -0.3323, -0.2239],
#           ...,
#           [-0.0624,  0.5362, -0.5600,  ...,  0.7437, -0.3323, -0.2239],
#           [-0.0624,  0.5362, -0.5600,  ...,  0.7437, -0.3323, -0.2239],
#           [-0.0624,  0.5362, -0.5600,  ...,  0.7437, -0.3323, -0.2239]]]],
#        grad_fn=<MeanBackward1>)
# tensor([[[[ 0.4851,  0.7255,  0.5718,  ..., -0.0023,  0.5366, -0.8882],
#           [ 0.4851,  0.7255,  0.5718,  ..., -0.0023,  0.5366, -0.8882],
#           [ 0.4851,  0.7255,  0.5718,  ..., -0.0023,  0.5366, -0.8882],
#           ...,
#           [ 0.4851,  0.7255,  0.5718,  ..., -0.0023,  0.5366, -0.8882],
#           [ 0.4851,  0.7255,  0.5718,  ..., -0.0023,  0.5366, -0.8882],
#           [ 0.4851,  0.7255,  0.5718,  ..., -0.0023,  0.5366, -0.8882]]]],
#        grad_fn=<CompiledFunctionBackward>)

# ...
# AssertionError: Tensor-likes are not equal!
# 
# Mismatched elements: 698 / 30498 (2.3%)
# Greatest absolute difference: 1 at index (0, 0, 0, 28, 10)
# Greatest relative difference: inf at index (0, 0, 0, 28, 10)