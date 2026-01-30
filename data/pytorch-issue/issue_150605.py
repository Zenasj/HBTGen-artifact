import torch
from optimum.quanto import ActivationQBytesTensor, absmax_scale, qint8, quantize_activation


device = torch.device("cpu")
input_shape = (10, 32, 32)
a = torch.randn(input_shape).to(device)

def f(x, dtype):
    return x.to(dtype)

scale = absmax_scale(a)
qa = quantize_activation(a, qtype=qint8, scale=scale)
compile_f = torch.compile(f)
cqa = compile_f(qa, torch.float16)
assert isinstance(cqa, ActivationQBytesTensor)
assert cqa.qtype == qint8
assert cqa._scale.dtype == torch.float16