import torch.nn as nn
import torch.nn.functional as F

import torch, torchvision
from typing import List, Dict
import torchvision.transforms.functional as F

class WrapPerspectiveCrop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, points: List[List[int]]):
        size_points = [[0,0], [inputs.shape[2],0] , [inputs.shape[2],inputs.shape[1]], [0,inputs.shape[1]]]
        inputs = F.perspective(inputs, points, size_points)
        return inputs

    
crop = WrapPerspectiveCrop()
scripted_model = torch.jit.script(crop)
scripted_model.save("wrap_perspective.pt")


import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

optimized_scripted_module=optimize_for_mobile(scripted_model)
optimized_scripted_module._save_for_lite_interpreter("wrap_perspective.ptl")

import cv2
import torch , torchvision
import numpy as np


model = torch.jit.load('my_model.ptl')

img = cv2.imread("sample.jpg")
img_tensor = torch.as_tensor(np.expand_dims(img.astype("float32").transpose(2, 0, 1) , axis = 0))
outputs = model(img_tensor);

print(outputs)