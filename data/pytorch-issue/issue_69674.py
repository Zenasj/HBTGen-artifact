with open("model_final_3c3198.pkl", "rb") as f:
    data = pickle.load(f)

data['model']['roi_heads.mask_head.coarse_head.fc1.bias'] = data['model']['roi_heads.mask_coarse_head.coarse_mask_fc1.bias']
data['model']['roi_heads.mask_head.coarse_head.fc1.weight'] = data['model']['roi_heads.mask_coarse_head.coarse_mask_fc1.weight']
data['model']['roi_heads.mask_head.coarse_head.fc2.bias'] = data['model']['roi_heads.mask_coarse_head.coarse_mask_fc2.bias']
data['model']['roi_heads.mask_head.coarse_head.fc2.weight'] = data['model']['roi_heads.mask_coarse_head.coarse_mask_fc2.weight']
data['model']['roi_heads.mask_head.coarse_head.prediction.bias'] = data['model']['roi_heads.mask_coarse_head.prediction.bias']
data['model']['roi_heads.mask_head.coarse_head.prediction.weight'] = data['model']['roi_heads.mask_coarse_head.prediction.weight']
data['model']['roi_heads.mask_head.coarse_head.reduce_spatial_dim_conv.bias'] = data['model']['roi_heads.mask_coarse_head.reduce_spatial_dim_conv.bias']
data['model']['roi_heads.mask_head.coarse_head.reduce_spatial_dim_conv.weight'] = data['model']['roi_heads.mask_coarse_head.reduce_spatial_dim_conv.weight']

del data['model']['roi_heads.mask_coarse_head.coarse_mask_fc1.bias']
del data['model']['roi_heads.mask_coarse_head.coarse_mask_fc1.weight']
del data['model']['roi_heads.mask_coarse_head.coarse_mask_fc2.bias']
del data['model']['roi_heads.mask_coarse_head.coarse_mask_fc2.weight']
del data['model']['roi_heads.mask_coarse_head.prediction.bias']
del data['model']['roi_heads.mask_coarse_head.prediction.weight']
del data['model']['roi_heads.mask_coarse_head.reduce_spatial_dim_conv.bias']
del data['model']['roi_heads.mask_coarse_head.reduce_spatial_dim_conv.weight']


data['model']['roi_heads.mask_head.point_head.fc1.bias'] = data['model']['roi_heads.mask_point_head.fc1.bias']
data['model']['roi_heads.mask_head.point_head.fc1.weight'] = data['model']['roi_heads.mask_point_head.fc1.weight']
data['model']['roi_heads.mask_head.point_head.fc2.bias'] = data['model']['roi_heads.mask_point_head.fc2.bias']
data['model']['roi_heads.mask_head.point_head.fc2.weight'] = data['model']['roi_heads.mask_point_head.fc2.weight']
data['model']['roi_heads.mask_head.point_head.fc3.bias'] = data['model']['roi_heads.mask_point_head.fc3.bias']
data['model']['roi_heads.mask_head.point_head.fc3.weight'] = data['model']['roi_heads.mask_point_head.fc3.weight']
data['model']['roi_heads.mask_head.point_head.predictor.bias'] = data['model']['roi_heads.mask_point_head.predictor.bias']
data['model']['roi_heads.mask_head.point_head.predictor.weight'] = data['model']['roi_heads.mask_point_head.predictor.weight']

del data['model']['roi_heads.mask_point_head.fc1.bias']
del data['model']['roi_heads.mask_point_head.fc1.weight']
del data['model']['roi_heads.mask_point_head.fc2.bias']
del data['model']['roi_heads.mask_point_head.fc2.weight']
del data['model']['roi_heads.mask_point_head.fc3.bias']
del data['model']['roi_heads.mask_point_head.fc3.weight']
del data['model']['roi_heads.mask_point_head.predictor.bias']
del data['model']['roi_heads.mask_point_head.predictor.weight']

with open("model_final_3c3198_hacked.pkl", "wb") as g:
    pickle.dump(data, g)

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.export import add_export_config, TracingAdapter
from detectron2.modeling import build_model
from projects.PointRend import point_rend


# Set cfg
cfg = get_cfg()
cfg.DATALOADER.NUM_WORKERS = 0
cfg = add_export_config(cfg)
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("/pytorch_repros/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.MODEL.WEIGHTS = "/pytorch_repros/model_final_3c3198_hacked.pkl"

# cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.freeze()

# create a torch model
torch_model = build_model(cfg)
DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)

inference = None
from detectron2.modeling import GeneralizedRCNN
if isinstance(torch_model, GeneralizedRCNN):
    def inference(model, inputs):
        # use do_postprocess=False so it returns ROI mask
        inst = model.inference(inputs, do_postprocess=False)[0]
        return [{"instances": inst}]

# get a sample data
data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
first_batch = next(iter(data_loader))

# convert and save caffe2 model
adapter_model = TracingAdapter(torch_model, first_batch, inference, allow_non_tensor=True)
adapter_model.eval()
torch.onnx.export(adapter_model,
                  adapter_model.flattened_inputs,
                  '_caffe2_detectron2_error_onnx_aten_fallback.onnx',
                  opset_version=16,
                  verbose=True,
                  training=torch.onnx.TrainingMode.EVAL,
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)