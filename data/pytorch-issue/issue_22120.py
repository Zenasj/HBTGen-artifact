weights_path = '/home/ws/DL/yolact/weights/yolact_im700_54_800000.pth'

import torch
import torch.onnx
import yolact
import torchvision

model = yolact.Yolact()

# state_dict = torch.load(weights_path)
# model.load_state_dict(state_dict)

model.load_weights(weights_path)

dummy_input = torch.randn(1, 3, 640, 480)

torch.onnx.export(model, dummy_input, "onnx_model_name.onnx")