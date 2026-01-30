import torch.nn as nn

r"""
Check torch export
"""
from typing import Tuple
import torch
import cv2

from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
image = cv2.imread("./notebooks/images/truck.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


predictor = SamPredictor(sam)


torch_img = predictor.transform.apply_image(image)
torch_img = torch.as_tensor(torch_img)
torch_img = torch_img.permute(2, 0, 1).contiguous()[None, :, :, :]

img_shape = (torch_img.shape[0], torch_img.shape[1])
predictor.set_torch_image(torch_img, img_shape)


class MyModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, torch_img: torch.Tensor, img_shape: Tuple[int, ...]):
        return self.module.set_torch_image(torch_img, img_shape)


exported_set_torch_img = torch.export.export(
    MyModule(predictor), (torch_img, img_shape), kwargs=None, constraints=None
)

torch.export.save(exported_set_torch_img, "/tmp/graph.pt2")

module = MyModule(predictor)
so_path = torch._export.aot_compile(
        module,
        (torch_img, img_shape))