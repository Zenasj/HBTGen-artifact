import torch
import torchvision
from torch.fx.experimental.proxy_tensor import make_fx
from torch._dynamo.backends.common import aot_autograd

def backend(gm, inputs):
    return make_fx(gm, tracing_mode="fake", _allow_non_fake_inputs=True)(*inputs)
backend=aot_autograd(fw_compiler=backend)

m = torchvision.models.resnet18()
m = torch.compile(m, backend=backend, dynamic=False)
x = m(torch.randn(1, 3, 224, 224))