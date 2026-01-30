import torch.nn as nn
import torchvision

import torch
from torchvision.models import resnet18

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = resnet18(**kwargs)

    @torch.no_grad()
    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        return features
    
enc = Encoder(pretrained=True).eval().to("cuda")
input_image_pytorch = torch.randn((1, 3, 480, 768), requires_grad=False).to("cuda").detach()
scripted_enc = torch.jit.script(enc, input_image_pytorch)

# Export the model
torch.onnx.export(scripted_enc,               # model being run
                  input_image_pytorch,                         # model input (or a tuple for multiple inputs)
                  "encoder.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                #   opset_version=17,          # the ONNX version to export the model to
                  # do_constant_folding=True,  # whether to execute constant folding for optimization. buggy in torch == 1.12
                  input_names = ['input'],   # the model's input names
                  output_names = ['f1', 'f2', 'f3', 'f4', 'f5'], # the model's output names
                  )