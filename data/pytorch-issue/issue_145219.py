import torch.nn as nn

import torch


def fn():
    v4_0 = torch.nn.Parameter(torch.randn([8, 1, 4, 1], dtype=torch.float32), requires_grad=True)
    v5_0 = torch.nn.Parameter(torch.empty([1, 1, 4, 1], dtype=torch.float32), requires_grad=True)
    v6_0 = torch.cat((v4_0, v5_0), dim=0)
    v6_0_flat = v6_0.view(-1, 1)  # 展平并调整形状
    linear_layer = torch.nn.Linear(in_features=1, out_features=43, bias=True)
    v2_0 = linear_layer(v6_0_flat)
    v2_0_unsqueezed = v2_0.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度以满足 MaxPool2d 的输入要求
    maxpool_layer = torch.nn.MaxPool2d(kernel_size=(2, 42), stride=2, padding=0, dilation=1, ceil_mode=False)
    v1_0 = maxpool_layer(v2_0_unsqueezed)
    batchnorm_layer = torch.nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    v0_0 = batchnorm_layer(v2_0_unsqueezed)
    return v1_0, v0_0


ret_eager = fn()
compiled = torch.compile(fn)
ret_compiled = compiled()

# assert torch.allclose(ret_eager[0], ret_compiled[0]), '\n'.join(map(str, ["", ret_eager[0], ret_compiled[0]]))
# assert torch.allclose(ret_eager[1], ret_compiled[1]), '\n'.join(map(str, ["", ret_eager[1], ret_compiled[1]]))

torch.testing.assert_close(ret_eager[0], ret_compiled[0])
# OUTPUT:
# AssertionError: Tensor-likes are not close!
#
# Mismatched elements: 18 / 18 (100.0%)
# Greatest absolute difference: nan at index (0, 0, 16, 0) (up to 1e-05 allowed)
# Greatest relative difference: nan at index (0, 0, 16, 0) (up to 1.3e-06 allowed)

torch.testing.assert_close(ret_eager[1], ret_compiled[1])
# OUTPUT:
# AssertionError: Tensor-likes are not close!
#
# Mismatched elements: 1548 / 1548 (100.0%)
# Greatest absolute difference: nan at index (0, 0, 0, 0) (up to 1e-05 allowed)
# Greatest relative difference: nan at index (0, 0, 0, 0) (up to 1.3e-06 allowed)