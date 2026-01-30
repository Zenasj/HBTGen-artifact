import numpy as np

import torch
import torchvision
from typing import Optional

@torch.jit.script
def friendly_ts_xywh2xyxy(x):
    """Friendly Torchscript function version of xywh2xyxy. Mark with # CHANGED the lines changed from the original version"""
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right

    # CHANGED TS doens't support np.zeros_like
    # y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

#@torch.jit.script
def friendly_ts_non_max_suppression(prediction, conf_thres:float=0.1, iou_thres:float=0.6,
                                    merge:bool=False, classes:Optional[torch.Tensor]=None, agnostic:bool=False,
                                    run_on_mobile:bool=False):
    """It's only a fragment"""
    
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    # REMOVED
    #t = time.time()

    # CHANGED type mismatch in scripting. The list is a list of tensors and not of Nones
    # output = [None] * prediction.shape[0]
    output = [torch.empty_like(prediction)]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        # CHANGED use friendly TS version of xywh2xyxy
        print('x.requires_grad=', x.requires_grad)
        box = friendly_ts_xywh2xyxy(x[:, :4])# xywh2xyxy(x[:, :4])
        print('box.requires_grad=', box.requires_grad)
        # Detections matrix nx6 (xyxy, conf, cls)

    return output

tt = torch.rand([1, 20, 85])
ts_nms = torch.jit.script(friendly_ts_non_max_suppression)