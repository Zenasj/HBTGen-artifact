python
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_faster_mobilenet_v2():
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    num_classes = 47
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# export to ONNX
batch_size = 1
# load model
model = create_faster_mobilenet_v2()

# Dummy input to the model
x = torch.randn(batch_size, 3, 800, 800)

# set the model to inference mode
model.eval()

# running dummy image
torch_out = model(x)

print('Model outputs: ', torch_out[0]['boxes'].shape, torch_out[0]['labels'].shape, torch_out[0]['scores'].shape)
print('Model output boxes: ', torch_out[0]['boxes'])
print('Model output labels: ', torch_out[0]['labels'])
print('Model output scores: ', torch_out[0]['scores'])

# Export the model
torch.onnx.export(model,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  "fasterRCNN-231120-694-46C-batch1.onnx",  # model path + name
                  # export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,  # the ONNX version to export the model to
                  # do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                            'output': {0: 'batch_size'}
                                }
                  )