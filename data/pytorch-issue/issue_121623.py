import torch.nn as nn
import torch.nn.functional as F

python
import os, sys, time
import numpy as np

import torch, torch.nn as nn
from torchvision import transforms
from torchsummary import summary


data=torch.zeros((1,6,640,640))
d1=torch.ones((1,3,640,640))
d2=d1+1
data[:,:3,:,:]=d1
data[:,3:,:,:]=d2

# print(data.shape)

# seprator layer
class seprator_layers(nn.Module):
    def __init__(self) -> None:
        super(seprator_layers,self).__init__()
    
    def forward(self,x):
        img1,img2=x[:,:3,:,:],x[:,3:,:,:]
        return img1,img2

# custom classification model 
class classify(nn.Module):
    def __init__(self) -> None:
        super(classify,self).__init__()
        self.seprate=seprator_layers()
        # self.conv=nn.Conv2d(6,8,3,2)
        # self.conv2=nn.Conv2d(8,16,3,2)
        self.vgg11=torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        self.vgg19=torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)

        self.transform=transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        pass

    def forward(self,x):
        img1,img2=self.seprate(x)

        img1=self.transform(img1)
        img2=self.transform(img2)
        
        x1=self.vgg11(img1)
        x2=self.vgg19(img2)


        return x1,x2
        
# load model
model=classify()
model.eval()

## no error
# output=model(data)
# x1,x2=output
# print(f"shape of x1: {x1.shape}")
# print(f"shape of x2: {x2.shape}")

# save .onnx file           [error]
dummpy_input=torch.randn(1,6,640,640)
torch.onnx.export(model,dummpy_input,"./simo_onnx.onnx")

import argparse
from typing import Dict, Any
from collections import OrderedDict
import torchvision.transforms.functional as F
import torch
from torch import nn
import onnx
from onnx import helper
import onnxruntime
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'TEED'))
from TEED.ted import TED


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    return parser


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class AdaptedTeed(nn.Module):

    def __init__(self):
        super().__init__()
        self.ted = TED()
        self.register_buffer('preprocess_mean', torch.FloatTensor([104.007, 116.669, 122.679]).reshape(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.to(torch.float32)
        N, C, H, W = x.shape
        x = x - self.preprocess_mean
        x = F.resize(x, (H // 16 * 16, W // 16 * 16))
        x = self.ted(x)
        x = torch.sigmoid(x[-1])
        x = F.resize(x, (H, W))
        x_max = 255
        x_min = 0
        epsilon = 1e-12
        x = (x - torch.min(x)) * (x_max - x_min) / \
            ((torch.max(x) - torch.min(x)) + epsilon) + x_min
        # x = x.to(torch.uint8)
        x = x / 255
        return x

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        new_dict = OrderedDict([('ted.' + k, v) for k, v in state_dict.items()])
        return super().load_state_dict(new_dict, strict, assign)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    device='cuda'
    with torch.no_grad():
        model = AdaptedTeed().to(device)
        model.eval()
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        inp = torch.randn(1, 3, 2816, 2844).to(device)
        inp = torch.clamp(inp, 0, 1) * 255
        # breakpoint()
        output = model(inp)
        # breakpoint()

    torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  args.save_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=17,          # 17 the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size', 2: 'height', 3: 'width'},    # variable length axes
                                'output' : {0 : 'batch_size', 2: 'height', 3: 'width'}})
    
    onnx_model = onnx.load(args.save_path)
    onnx.checker.check_model(onnx_model)  # Check graph validity

    metadata = {
        "model_author": "Jose Pérez Cano",
        "model_license": "MIT",
        "date": "2024-04-08",
        "description": "The input must be of shape NCHW with values ranging 0-1 of type float32. Output is of shape HW, range 0-1 and type float32.",
    }
    helper.set_model_props(onnx_model, metadata)
    onnx.save(onnx_model, args.save_path)

    # Options: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
    ort_session = onnxruntime.InferenceSession(args.save_path, providers=['CPUExecutionProvider'])  

    # compute ONNX Runtime output prediction
    ort_inputs = {'input': to_numpy(inp)}
    ort_outs = ort_session.run(['output'], ort_inputs)
    
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=0.02, atol=1)



if __name__=='__main__':
    main()

inp = torch.randn(1, 3, 1024, 1024).to(device)