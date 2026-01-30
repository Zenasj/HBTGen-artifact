# Test with resnet
import torchvision
import torch

resnet = torchvision.models.resnet18()
resnet.to(0)

torch.onnx.export(resnet, # model
                  input_img, # input (can be a tuple of multiple inputs)
                  "resnet.onnx", # filepath
                  export_params=True) # stored weights in ONNX file