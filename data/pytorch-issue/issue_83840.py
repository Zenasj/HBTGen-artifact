import torch.nn as nn

py
torch.nn.functional.interpolate(x, size=[224, 224], mode="bilinear", align_corners=False, antialias=antialias)

py
import torch
from time import time

def bench(f, inp, num_exp=1000, num_prime=10):
    for _ in range(num_prime):
        f(inp)

    times = []
    for _ in range(num_exp):
        start = time()
        f(inp)
        end = time()
        times.append((end - start))
    
    median = torch.median(torch.tensor(times))
    print(f"Median over {num_exp} exp= {median * 1e6 :.1f} μs")
    return median


def interpolate(x, antialias):
    torch.nn.functional.interpolate(x, size=[224, 224], mode="bilinear", align_corners=False, antialias=antialias)


tensor_img = torch.randint(0, 256, (3, 64, 64), dtype=torch.float32)
assert tensor_img.stride() == (4096, 64, 1)
tensor_img = tensor_img[None, :, :, :]  # add batch dim for call to interpolate()

bench(lambda x: interpolate(x, antialias=True), tensor_img)  # 175.7 μs
bench(lambda x: interpolate(x, antialias=False), tensor_img)  # 129.7 μs

py
tensor_img = torch.randint(0, 256, (3, 64, 64), dtype=torch.float32)
assert tensor_img.stride() == (4096, 64, 1)
tensor_img = tensor_img.as_strided(tensor_img.size(), stride=(1, 192, 3))
tensor_img = tensor_img[None, :, :, :]  # add batch dim for call to interpolate()

bench(lambda x: interpolate(x, antialias=True), tensor_img)  # 176.4 μs
bench(lambda x: interpolate(x, antialias=False), tensor_img)  # 1471.8 μs  8X slower!!!

py
import torch
import torchvision.transforms as T
from torchvision.io import read_file, decode_jpeg
from PIL import Image
import matplotlib.pyplot as plt

file = "test/assets/encode_jpeg/grace_hopper_517x606.jpg"

from_pil = T.ToTensor()(Image.open(file))
assert from_pil.stride() == (313302, 517, 1)
from_decode = T.ConvertImageDtype(torch.float)(decode_jpeg(read_file(file)))
assert from_decode.stride() == (1, 1551, 3)

def interpolate(x, antialias):
    return torch.nn.functional.interpolate(x, size=[1000, 1000], mode="bilinear", align_corners=False, antialias=antialias)
    
    
for antialias in (True, False):
    for img in (from_pil, from_decode):
        img = img[None, :, :, :]  # add batch dim for call to interpolate()

        out = interpolate(img, antialias=antialias)
        fig = plt.figure()
        plt.imshow(out[0].permute(1, 2, 0))
plt.show()