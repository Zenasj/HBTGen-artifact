import torch.nn as nn
import torch.nn.functional as F

import cv2
import torch
import torchvision.transforms.functional as F

def test_rescale_with_padding():
    img = torch.tensor(
        cv2.imread("tests/data/2022-11-10T02-18-31-036Z-1066.jpg"), dtype=torch.float32
    ).permute(2, 0, 1)[None, :]
    result_img = RescaleWithPadding(512, 512)(img)
    module = torch.jit.script(RescaleWithPadding(512, 512))
    torch.onnx.export(
        model=module,
        args=(img,),
        f="transforms.onnx",
        export_params=True,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"finished: {result_img.shape}")

class RescaleWithPadding(torch.nn.Module):
    def __init__(self, height: int, width: int, padding_value: int = 0):
        super(RescaleWithPadding, self).__init__()
        self.height = height
        self.width = width
        self.padding_value = padding_value
        self.max_size = max(height, width)
        self.interpolation = F.InterpolationMode.BILINEAR

    def forward(self, img: torch.Tensor):
        b, c, image_height, image_width = img.shape
        smaller_edge_size = min(image_height, image_width)
        img = F.resize(
            img=img,
            size=[smaller_edge_size],
            interpolation=self.interpolation,
            max_size=self.max_size,
        )
        return img