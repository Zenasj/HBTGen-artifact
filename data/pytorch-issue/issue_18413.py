import torch
import torch.nn as nn
import numpy as np

def torch_pad_reflect(image: torch.Tensor, paddings: Sequence[int]) -> torch.Tensor:
    paddings = np.array(paddings, dtype=int)

    assert np.all(np.array(image.shape[-2:]) > 1),  "Image shape should be more than 1 pixel"
    assert np.all(paddings >= 0), "Negative paddings not supported"

    while np.any(paddings):
        image_limits = np.repeat(image.shape[::-1][:len(paddings)//2], 2) - 1
        possible_paddings = np.minimum(paddings, image_limits)

        image = torch.nn.functional.pad(image, tuple(possible_paddings), mode='reflect')

        paddings = paddings - possible_paddings

    return image