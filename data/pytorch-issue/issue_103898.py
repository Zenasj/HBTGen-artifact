import torch.nn as nn
import numpy as np

py
from PIL import Image
from torchvision import transforms as T
import time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torchvision import models 
import torch
import torch_tensorrt

fcn = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights,pretrained=True).eval()
fcn = fcn.cuda()

def segment(net, path, show_orig=True, dev='cuda'):
  img = Image.open(path)
 
  if show_orig: plt.imshow(img); plt.axis('off'); plt.show()

  trf = T.Compose([T.Resize(640), 
                  #  T.CenterCrop(640), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
  inp = trf(img).unsqueeze(0).to(dev)
  out = net.to(dev)(inp)
  print("Input Size: ", inp.shape)
  print("Output Size: ", out['out'].shape) 

fcn_traced = torch.jit.trace(fcn, [torch.randn((1,3,640,1000)).to("cuda")],strict=False)

fcn_fp32 = torch_tensorrt.compile(fcn_traced, inputs = [torch_tensorrt.Input(
    min_shape=(1, 3, 640, 224),
    opt_shape=(16, 3, 640, 1000),
    max_shape=(16, 3, 640, 1920), dtype=torch.float32)],
    enabled_precisions = torch.float32, # Run with FP32
    workspace_size = 1 << 22
)