import torch
from torch import Tensor

torch._dynamo.config.capture_dynamic_output_shape_ops = True

@torch.library.custom_op("mylib::mk_image", mutates_args=("decoder",), device_types=["cpu"])
def mk_image(decoder: Tensor) -> Tensor:
    return torch.randn(2,3,4,5)

@torch.library.register_fake("mylib::mk_image")
def _(decoder: Tensor) -> Tensor:
    image_size = [torch.library.get_ctx().new_dynamic_size() for _ in range(4)]
    return torch.empty(image_size)

@torch.compile(fullgraph=True)
def f(x):
    return torch.ops.mylib.mk_image.default(x)

x = torch.zeros(100, dtype=torch.int64)
f(x)