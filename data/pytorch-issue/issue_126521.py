import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torch

if __name__ == "__main__":
    print(torch.__version__)
    qconv_module = nnq.Conv2d(3, 1, 3)
    conv_module = nn.Conv2d(3, 1, 3)

    in_channels = 3
    out_channels = 1
    inp_scale = 0.007812488358467817
    inp_zero_point = 128
    X = torch.Tensor([[[[61., 61., 59.],
        [70., 69., 65.],
        [79., 81., 75.]],

        [[61., 61., 59.],
        [70., 69., 65.],
        [79., 81., 75.]],

        [[61., 61., 59.],
        [70., 69., 65.],
        [79., 81., 75.]]]]) * inp_scale

    X_q = torch.quantize_per_tensor(X, 
        inp_scale, inp_zero_point, torch.quint8)

    W_scale = 0.046079955995082855
    W_zero_point = 0
    W = torch.Tensor([[[[  12,  -89,   17],
        [ -24, -116, -102],
        [ -18,   78,  -83]],

        [[  -8,  127,   92],
        [  15,  -11,  127],
        [  43,  -45,   -5]],

        [[ -44,   40,  -35],
        [ -60,  -35,  -44],
        [  90,  -36,   85]]]]) * W_scale
    W_q = torch.quantize_per_tensor(W, W_scale, W_zero_point, torch.qint8)
    b = torch.Tensor([1.193039894104004]).float()

    example_input = [X, ]
    example_input_q = [X_q, ]


    # Make sure the weight shape is correct
    Y_scale = 0.09463921934366226
    Y_zero_point = 0
    qconv_module.set_weight_bias(W_q, b)
    qconv_module.scale = Y_scale
    qconv_module.zero_point = Y_zero_point

    raw_conv_module = conv_module
    raw_conv_module.weight.data = W
    raw_conv_module.bias.data = b

    # Test forward
    Y_exp = conv_module(*example_input)
    Y_exp = torch.quantize_per_tensor(
        Y_exp, scale=Y_scale, zero_point=Y_zero_point, dtype=torch.quint8)
    Y_act = qconv_module(*example_input_q)

    print(Y_act.dequantize().item(), Y_exp.item())