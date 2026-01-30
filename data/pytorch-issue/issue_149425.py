py
import torch

img = torch.ones(3, 64, 64)
img *= torch.linspace(0, 1, steps=64) * torch.linspace(0, 1, steps=64).unsqueeze(-1)

def f(img):
    return crop(img, (10, 10, 50, 50))

from typing import Sequence

@torch.library.custom_op("mylib::crop", mutates_args=())
def crop(pic: torch.Tensor, box: Sequence[int]) -> torch.Tensor:
    channels = pic.shape[0]
    x0, y0, x1, y1 = box
    result = pic[:, y0:y1, x0:x1].permute(1, 2, 0).contiguous().permute(2, 0, 1)
    return result

@crop.register_fake
def _(pic, box):
    channels = pic.shape[0]
    x0, y0, x1, y1 = box
    result = pic.new_empty(y1 - y0, x1 - x0, channels).permute(2, 0, 1)
    return result


@torch.compile(fullgraph=True)
def f(img):
    return crop(img, (10, 10, 50, 50))

cropped_img = f(img)
print(cropped_img.shape)
print(cropped_img.stride())

py
def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (3, 64, 64), (4096, 64, 1))
    # Topologically Sorted Source Nodes: [crop_default], Original ATen: [mylib.crop]
    buf0 = torch.ops.mylib.crop.default(arg0_1, [10, 10, 50, 50])
    del arg0_1
    buf1 = buf0
    assert_size_stride(buf1, (3, 40, 40), (1600, 40, 1))
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, 64, 64), (4096, 64, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)