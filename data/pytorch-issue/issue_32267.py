import torch.nn as nn
import torchvision

import torch

from PIL import Image
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.structures.bounding_box import BoxList

import numpy as np
import onnx
import onnxruntime as ort

config_file = "./configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.freeze()

# Create model
coco_demo = COCODemo(
    cfg,
    confidence_threshold=0.7,
    min_image_size=800,
)

for p in coco_demo.model.parameters():
    p.requires_grad_(False)

class Backbone(torch.nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

    def forward(self, image):
        image_list = ImageList(image.unsqueeze(0), [(image.size(-2), image.size(-1))])

        result = coco_demo.model.backbone(image_list.tensors)
        return result

backbone = Backbone()
backbone.eval()

from torchvision import transforms as T
from torchvision.transforms import functional as F
from demo.predictor import Resize

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = np.array(image)[:, :, [2, 1, 0]]
    return image

def build_transform(cfg):
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    transform = T.Compose(
        [
            T.ToPILImage(),
            Resize(min_size, max_size),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform

def transform_image(cfg, original_image):
    transforms = build_transform(cfg)
    image = transforms(original_image)
    from maskrcnn_benchmark.structures.image_list import to_image_list
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
    image_list = image_list.to(torch.device(cfg.MODEL.DEVICE))
    return (image_list.tensors[0], image.size(-1), image.size(-2))

original_image = load_image("./demo/sample.jpg")
from transform import transform_image
image, _, _ = transform_image(cfg, original_image)

expected_result = backbone(image)

BACKBONE_OPS10_PATH = "backbone_ops10.onnx"
BACKBONE_OPS11_PATH = "backbone_ops11.onnx"

torch.onnx.export(backbone, image, BACKBONE_OPS10_PATH,
                    verbose=False,
                    do_constant_folding=False,
                    opset_version=10, input_names=["i_image"])
torch.onnx.export(backbone, image, BACKBONE_OPS11_PATH,
                    verbose=False,
                    do_constant_folding=False,
                    opset_version=11, input_names=["i_image"])

ort_session = ort.InferenceSession(BACKBONE_OPS10_PATH)
input_feed = {ort_session.get_inputs()[0].name: image.numpy()}
backbone_ops10_result = ort_session.run(None, input_feed)

ort_session = ort.InferenceSession(BACKBONE_OPS11_PATH)
input_feed = {ort_session.get_inputs()[0].name: image.numpy()}
backbone_ops11_result = ort_session.run(None, input_feed)