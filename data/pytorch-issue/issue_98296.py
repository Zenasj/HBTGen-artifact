import torch.nn as nn

import torch
from torch import nn
from ultralytics.yolo.utils.ops import non_max_suppression
from ultralytics.yolo.v8.detect import DetectionPredictor

class PostProcessingModule(nn.Module, DetectionPredictor):
    def forward(self, yolo_results, iou_threshold, score_threshold):
        return non_max_suppression(
            yolo_results, iou_threshold, score_threshold
        )

if __name__ == '__main__':

    yolo_results = torch.rand([1, 14, 1000]).type(torch.float32)
    iou_threshold = 0.5
    score_threshold = 0.5
    
    t_model = PostProcessingModule()
    torch.onnx.export(
        t_model,
        (yolo_results, iou_threshold, score_threshold),
        "NMS_after.onnx",
        input_names=["yolo_results", "iou_threshold", "score_threshold"],
        output_names=["yolo_results_filtered"],
    )

multi_label &= nc > 1

multi_label = multilabel & (nc > 1)