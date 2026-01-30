import os
import sys 

from infer_effdet.py import EfficientDet # the file we copied from the gist

import torch
import onnx
import numpy as np
import cv2

model = EfficientDet(
    num_classes=6
)

model = model.eval()

dummy = torch.randn(1, 3, 512, 512)
model.backbone_net.model.set_swish(memory_efficient=False)

torch.onnx.export(
    model, 
    dummy, 
    "trial6.onnx", 
    export_params=True, 
    verbose=False,
    opset_version=11, 
)

model.backbone_net.model.set_swish(memory_efficient=True)

def forward(self, x):


        is_training = False
        imgs = x 

        c3, c4, c5 = self.backbone_net(imgs)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)
        p6 = self.conv6(c5)
        p7 = self.conv7(p6)

        features = [p3, p4, p5, p6, p7]
        features = self.bifpn(features)

        regression = torch.cat([self.regressor(feature) for feature in features], dim=1)
        classification = torch.cat([self.classifier(feature) for feature in features], dim=1)
        anchors = self.anchors(imgs)


            
        ret_val =  _scriptfied_func(anchors, regression, classification, imgs)
        
        #a = ret_val[0, :, 0]
        #b = ret_val[1, :, 0]
        #c = ret_val[2, :]
        #print(a)
        #print(ret_val.shape)
        sample_ret =  torch.stack([
        torch.zeros(64, 4), torch.zeros(64, 4), torch.zeros(64, 4)], dim=0)
        #print(sample_ret.shape) 
        return ret_val