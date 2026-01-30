import torch.nn as nn
import numpy as np
import random

img = np.random.randint(0, 256, size=(479, 640, 3), dtype=np.uint8)
task = "panoptic"

with torch.no_grad():
    original_image = img[:, :, ::-1]
    height, width = img.shape[:2]
    image = predictor.aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    
    inputs = {"image": image, "height": height, "width": width, "task": task}

    ##### SUCCEEDS
    predictor.model([inputs])

    ##### FAILS
    optimized = torch.compile(predictor.model, backend="eager")
    optimized([inputs])

import torch
import open_clip

class SampleModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def forward(self, x):
        text = self.tokenizer(x)
        return text

"""
Code adapted from:
https://github.com/mlfoundations/open_clip/blob/fb72f4db1b17133befd6c67c9cf32a533b85a321/README.md?plain=1#L62-L71
"""
model = SampleModule().eval().cuda()
input_ = ["a diagram", "a dog", "a cat"]

# Passes
model(input_)


# Fails
compiled = torch.compile(model, backend="eager")
compiled(input_)