from models import *
from datasets import letterbox #*
from utils import *
import numpy as np
import sys, random
import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import io
import torch.nn as nn
from torch.autograd import Variable
import cv2
import os
import logging

img_size=(416, 416)
cfg = 'yolov3-spp.cfg'
imgsz = img_size

import torch
import torch.onnx


model_pt_path = "last.pt"
model_def_path = "models.py"

model = Darknet(cfg, imgsz)
model.load_state_dict(torch.load(model_pt_path),strict=False)

dummy_input = torch.randn(1,3,416,416)

torch.onnx.export(model, dummy_input, "SL-PMH.onnx")