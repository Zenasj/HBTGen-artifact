import torch.nn as nn

py
import random

import torch
import pytest
import numpy as np
from PIL import Image
from torch.nn.functional import interpolate

@pytest.mark.parametrize("C", (1, 3, 6))
@pytest.mark.parametrize("batch_size", (1, 4))
@pytest.mark.parametrize("memory_format", (torch.contiguous_format, torch.channels_last, "strided", "cropped"))
@pytest.mark.parametrize("antialias", (True, False))
# @pytest.mark.parametrize("mode", ("bilinear", "bicubic",))
@pytest.mark.parametrize("mode", ("bicubic",))
@pytest.mark.parametrize("seed", range(100))
def test_resize(C, batch_size, memory_format, antialias, mode, seed):

    torch.manual_seed(seed)
    random.seed(seed)

    Hi = 2**random.randint(3, 10) + random.randint(0, 30)
    Wi = 2**random.randint(3, 10) + random.randint(0, 30)
    Ho = 2**random.randint(3, 10) + random.randint(0, 30)
    Wo = 2**random.randint(3, 10) + random.randint(0, 30)
    # print(Hi, Wi, Ho, Wo)

    img = torch.randint(0, 256, size=(batch_size, C, Hi, Wi), dtype=torch.uint8)

    if memory_format in (torch.contiguous_format, torch.channels_last):
        img = img.to(memory_format=memory_format, copy=True)
    elif memory_format == "strided":
        img = img[:, :, ::2, ::2]
    elif memory_format == "cropped":
        a = random.randint(1, Hi // 2)
        b = random.randint(Hi // 2 + 1, Hi)
        c = random.randint(1, Wi // 2)
        d = random.randint(Wi // 2 + 1, Wi)
        img = img[:, :, a:b, c:d]
    else:
        raise ValueError("Uh?")

    margin = 0
    img = img.clip(margin, 255 - margin)
    out_uint8 = interpolate(img, size=[Ho, Wo], mode=mode, antialias=antialias)

    if antialias and C == 3:
        out_pil_tensor = resize_with_pil(img, Wo, Ho, mode=mode, antialias=antialias)
        atol = {"bicubic": 2, "bilinear": 1}[mode]
        torch.testing.assert_close(out_uint8, out_pil_tensor, rtol=0, atol=atol)

    out_float = interpolate(img.to(torch.float), size=[Ho, Wo], mode=mode, antialias=antialias).round().clip(0, 255).to(torch.uint8)
    if mode == "bicubic":
        # Note: when antialias=False, we have to use much bigger tolerances than when antialias=True.
        # This is partially due to the fact that when False, the float path uses the -0.75 constant
        # while the uint8 path uses the -0.5 constant in the bicubic kernel (when True, they both use -0.5).
        # This difference in constants exists for historical reasons.
        # Should both paths use the -0.5 constant, we would have closer results and we would be able to lower the tolerances.
        diff = (out_float.float() - out_uint8.float()).abs()

        max_diff = 30 if antialias else 44
        assert diff.max() < max_diff

        threshold = 2
        percent = 3 if antialias else 40
        assert (diff > threshold).float().mean() < (percent / 100)

        threshold = 5
        percent = 1 if antialias else 20
        assert (diff > threshold).float().mean() < (percent / 100)

        mae = .4 if antialias else 3
        assert diff.mean() < mae
    else:
        torch.testing.assert_close(out_uint8, out_float, rtol=0, atol=1)



def resize_with_pil(batch, Wo, Ho, mode, antialias):
    resample = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}[mode]
    out_pil = [
        Image.fromarray(img.permute((1, 2, 0)).numpy()).resize((Wo, Ho), resample=resample)
        for img in batch
    ]
    out_pil_tensor = torch.cat(
        [
            torch.as_tensor(np.array(img, copy=True)).permute((2, 0, 1))[None]
            for img in out_pil
        ]
    )
    return out_pil_tensor