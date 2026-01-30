import torch

def forward():
    grad_out = torch.randn((1, 4, 64, 64))
    input_vec = torch.randn((1, 320, 64, 64))
    weight = torch.randn((4, 320, 3, 3))
    result0, _, _ = torch.ops.aten.convolution_backward(
        grad_out, input_vec, weight, bias_sizes=[4], stride=[1, 1],
        padding=[1, 1], dilation=[1, 1], transposed=False,
        output_padding=[0, 0], groups=1, output_mask=[False, True, True])
    return result0 is None

script_forward = torch.jit.script(forward)
print(script_forward.graph)
print(f"forward output: {forward()}")
print(f"script_forward output: {script_forward()}")