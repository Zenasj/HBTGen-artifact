import sys
import os
import torch
from backbone import EfficientDetBackbone
def pth_onnx(input_pth):

    outpath = os.path.join(os.path.dirname(input_pth), os.path.basename(input_pth)[:-3] + 'onnx')

    compound_coef = 0
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(input_pth, map_location='cpu'))
    model.requires_grad_(False)
    model.eval()

    # model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    # model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
    # model.requires_grad_(False)
    # model.eval()

    dummy_input = torch.ones(1, 3, 512, 512)
    # model.set_swish(memory_efficient=False)
    with torch.no_grad():
        torch.onnx.export(model, dummy_input, outpath, opset_version=11, verbose=False)

    return

if __name__ == '__main__':

    input_pth_path = r'weights/efficientdet-d0.pth'
    pth_onnx(input_pth_path)