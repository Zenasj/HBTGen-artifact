import torch.nn as nn
import random

from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import numpy as np
import argparse
import sys


def qconv_ref(
    inp, weight, bias, output, N, C, H, W, K, kernel, stride, pad, dilation, groups
):

    OH = (H + 2 * pad[0] - dilation[0] * (kernel[0] - 1) - 1) // stride[0] + 1
    OW = (W + 2 * pad[1] - dilation[1] * (kernel[1] - 1) - 1) // stride[1] + 1

    assert dilation[0] == 1
    assert dilation[1] == 1
    for i in range(N):
        for h in range(OH):
            for w in range(OW):
                for g in range(groups):
                    for m in range(K // groups):
                        sum = 0
                        for r in range(kernel[0]):
                            h_in = -pad[0] + h * stride[0] + r
                            for s in range(kernel[1]):
                                w_in = -pad[1] + w * stride[1] + s
                                for c in range(C // groups):
                                    a = 0
                                    if h_in < 0 or h_in >= H or w_in < 0 or w_in >= W:
                                        a = 0
                                    else:
                                        a = inp[i][g * (C // groups) + c][h_in][w_in]
                                    b = weight[g * (K // groups) + m][c][r][s]
                                    sum += a * b
                        output[i][g * (K // groups) + m][h][w] = (
                            sum + bias[g * (K // groups) + m]
                        )


def run_test(input_channels, cuda):
    batch_size = 1
    output_channels = 16
    height = 5
    width = 5
    groups = 2
    kernel_h = 1
    kernel_w = 1
    stride_h = 1
    stride_w = 1
    pad_h = 0
    pad_w = 0
    dilation_h = 1
    dilation_w = 1

    np.random.seed(0)

    W_value_min = 0
    W_value_max = 5

    W_init = torch.from_numpy(
        np.random.randint(
            W_value_min,
            W_value_max,
            (output_channels, input_channels // groups, kernel_h, kernel_w),
        )
    )

    # b_init = torch.from_numpy(np.random.randint(0, 10, (output_channels,)))
    b_init = torch.from_numpy(np.zeros((output_channels,)))

    conv_op = torch.nn.Conv2d(
        input_channels,
        output_channels,
        (kernel_h, kernel_w),
        (stride_h, stride_w),
        (pad_h, pad_w),
        (dilation_h, dilation_w),
        groups,
    )

    conv_op.weight = torch.nn.Parameter(
        W_init.to(dtype=torch.float), requires_grad=False
    )

    conv_op.bias = torch.nn.Parameter(b_init.to(dtype=torch.float), requires_grad=False)

    X_value_min = 0
    X_value_max = 5
    X_init = torch.from_numpy(
        np.random.randint(
            X_value_min, X_value_max, (batch_size, input_channels, height, width)
        )
    )

    if cuda:
        conv_op.cuda()
        X_init = X_init.cuda()
    result_ref = conv_op(X_init.to(dtype=torch.float))
    if cuda:
        result_ref = result_ref.cpu()

    result_local_ref = result_ref.clone()
    qconv_ref(
        X_init,
        W_init,
        b_init,
        result_local_ref,
        batch_size,
        input_channels,
        height,
        width,
        output_channels,
        [kernel_h, kernel_w],
        [stride_h, stride_w],
        [pad_h, pad_w],
        [dilation_h, dilation_w],
        groups,
    )
    # print(result_ref.permute([0, 2, 3, 1]))
    # print(result_local_ref.permute([0, 2, 3, 1]))

    np.testing.assert_equal(result_local_ref.numpy(), result_ref.numpy())


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        action='store_true',
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=6,
        help="Number of input channels. Must be a multiple of 2",
    )
    options = parser.parse_args(args)

    run_test(options.input_channels, options.cuda)


if __name__ == "__main__":
    main(sys.argv[1:])