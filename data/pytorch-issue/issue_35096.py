import torch.nn as nn
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx as tonnx

from os.path import isfile, isdir, join

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch

from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes

class ToOnnx(nn.Module):
    def __init__(self, model_name, ckpt_dir):
        super(ToOnnx, self).__init__()
        self.model_name = model_name
        self.ckpt_dir = ckpt_dir
        self.initCfg()
        self.predictor = DefaultPredictor(self.cfg)

        self.add_module("mask_rcnn", self.predictor.model)

    def initCfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model_name))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_name)  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 5400    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon)

        cfg.OUTPUT_DIR = self.ckpt_dir
        cfg.MODEL.WEIGHTS = join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model

        self.cfg = cfg
        

    def exportModel(self, model, input, input_names, output_names,
    output_dir="super_resolution.onnx", version=10, dynamic_axes={}):
        def _check_eval(module):
            assert not module.training

        # model.eval()
        # print(model.training)
        # model.apply(_check_eval)
        with torch.no_grad():
            with open(output_dir, 'w') as f:
                tonnx.export(
                    model, ## Model being run
                    input,
                    "mask_rcnn.onnx",
                    verbose=False,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    use_external_data_format=True,
                    operator_export_type=OperatorExportTypes.ONNX_ATEN
                )


    def forward(self, original_image):

        original_image = np.asarray(original_image)

        out = self.predictor(original_image)
        predictions = out['instances'].to('cpu')
        img_height, img_width = predictions.image_size

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None


        if predictions.has("pred_masks"):
            masks = torch.tensor(predictions.pred_masks, dtype=int).detach()
            # masks = [GenericMask(x, img_height, img_width) for x in masks]
        else:
            masks = None

        
        return {
            'pred_boxes' : boxes.tensor,
            'scores' : scores,
            'pred_classes' : classes,
            'pred_masks' : masks
        }