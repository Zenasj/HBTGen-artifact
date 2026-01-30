import torch
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dummpy_input = torch.randn(1, 3, 800, 1333)
torch.onnx.export(model, dummpy_input, "FasterRCNN.onnx", verbose=True, input_names=['image_in'],
                  output_names = ['image_out'])