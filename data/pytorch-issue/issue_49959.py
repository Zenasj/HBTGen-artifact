import torch.nn as nn

#!/usr/bin/env python3

import argparse
import sys
from typing import List

sys.path.append('.')

import torch
import cv2
import torchvision

import models
import models.yolo
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import check_img_size
from utils.datasets import letterbox


@torch.jit.script
def box_area(box):
    # box = 4xn
    return (box[2] - box[0]) * (box[3] - box[1])


@torch.jit.script
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


@torch.jit.script
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


@torch.jit.script
def loop_body(xi: int, x: torch.Tensor, multi_label: bool, xc: torch.Tensor,
        output: List[torch.Tensor], labels: torch.Tensor, nc: int,
        conf_thres: float, classes: torch.Tensor, agnostic: bool,
        iou_thres: float):
    max_wh = 4096
    max_det = 300
    # Apply constraints
    # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
    x = x[xc[xi]]  # confidence

    # Cat apriori labels if autolabelling
    if len(labels.size()) and labels and len(labels[xi]):
        l = labels[xi]
        v = torch.zeros((len(l), nc + 5), device=x.device)
        v[:, :4] = l[:, 1:5]  # box
        v[:, 4] = 1.0  # conf
        v[torch.arange(len(l)), l[:, 0].long() + 5] = 1.0  # cls
        x = torch.cat((x, v), 0)

    # If none remain process next image
    if not x.shape[0]:
        return

    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
        tmp = (x[:, 5:] > conf_thres).nonzero().T
        i, j = tmp[0], tmp[1]
        x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
    else:  # best class only
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

    # Filter by class
    if len(classes.size()) and classes:
        x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

    # Apply finite constraint
    # if not torch.isfinite(x).all():
    #     x = x[torch.isfinite(x).all(1)]

    # If none remain process next image
    if not x.shape[0]:
        return

    # Sort by confidence
    # x = x[x[:, 4].argsort(descending=True)]

    # Batched NMS
    c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]

    output[xi] = x[i]


@torch.jit.script
def non_max_suppression(prediction,
        conf_thres: float = 0.25, iou_thres: float = 0.45,
        classes: torch.Tensor = torch.tensor(0),
        agnostic: bool = False, labels: torch.Tensor = torch.tensor(0)):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        loop_body(xi=xi, x=x, multi_label=multi_label, xc=xc, output=output,
            labels=labels, classes=classes, agnostic=agnostic,
            iou_thres=iou_thres, conf_thres=conf_thres, nc=nc)

    return output


class CombinedModel(torch.nn.Module):
    def __init__(self, yolov5):
        super().__init__()
        self.yolov5 = yolov5

    def forward(self, x):
        pred = self.yolov5(x)
        return non_max_suppression(pred[0])


def parse_cmdline():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--img-size', nargs='+', type=int, default=[640, 640])
    p.add_argument('--test-image', required=True)
    p.add_argument('--output', required=True)
    return p.parse_args()


def load_model(path, size):
    model = attempt_load(path, map_location=torch.device('cpu'))
    gs = int(max(model.stride))  # grid size (max stride)
    size = [check_img_size(x, gs) for x in size]  # verify img_size are gs-multiples
    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, torch.nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, torch.nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = False
    return model, size


def load_test_image(filename, size):
    image = cv2.imread(filename)
    if list(image.shape[:2]) != list(size):
        print("Invalid image shape", image.shape, file=sys.stderr)
        sys.exit(1)
    image = letterbox(image, new_shape=640)[0]
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = torch.from_numpy(image.copy()).float() / 255.0

    return image.unsqueeze(0)


def main():
    cmdline = parse_cmdline()

    yolov5, img_size = load_model(cmdline.weights, cmdline.img_size)
    batch = load_test_image(cmdline.test_image, img_size)

    # This call is a must, it will change the model internal structure.
    y = yolov5(batch)
    print(y[0].shape)

    combined = CombinedModel(yolov5)

    traced = torch.jit.trace_module(combined, {'forward': batch})
    r = combined(batch)
    print(r[0].shape)

    torch.onnx.export(combined, batch, cmdline.output, opset_version=11)


if __name__ == '__main__':
    print(non_max_suppression.code)
    print(loop_body.code)
    main()

output = []
for i in range(prediction.shape[0]):
    output.append(torch.zeros((0, 6), device=prediction.device))