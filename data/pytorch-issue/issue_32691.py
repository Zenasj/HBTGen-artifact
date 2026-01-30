import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub 
from torch.nn.quantized import FloatFunctional 
from torchvision.models import quantization as quantized_models 

# convert this into a class 
class Conv_BN_RELU(nn.Sequential):
    def __init__(self,inp, oup, stride = 1 ):
        super(Conv_BN_RELU, self).__init__(
              nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
              nn.BatchNorm2d(oup),
              nn.ReLU()
                       )

class Conv_BN(nn.Sequential):
    def __init__(self,inp, oup, stride):
        super().__init__(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                       nn.BatchNorm2d(oup))

class Conv_BN1x1_RELU(nn.Sequential):
    def __init__(self,inp, oup, stride):
        super().__init__(nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
                       nn.BatchNorm2d(oup),
                       nn.ReLU())

class Conv_DW(nn.Sequential):
    def __init__(self,inp, oup, stride):
        super().__init__(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                       nn.BatchNorm2d(inp),
                       nn.ReLU(),
                       nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                       nn.BatchNorm2d(oup),
                       nn.ReLU()
        )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = Conv_BN(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = Conv_BN_RELU(in_channel, out_channel//4, stride=1)
        self.conv5X5_2 = Conv_BN(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = Conv_BN_RELU(out_channel//4, out_channel//4, stride=1)
        self.conv7x7_3 = Conv_BN(out_channel//4, out_channel//4, stride=1)

        # define our quantizer and dequantizer attributes 
        # to be used in our forward pass
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, input):

        # quantize the input 
        input = self.quant(input)    
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        # note: should we use f_cat = FloatFunctional() ? 
        # out = f_cat.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        # dequantize our output before returning it
        out = self.dequant(out)
        return out

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == Conv_BN_RELU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            elif type(m) == Conv_BN:
                torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = Conv_BN1x1_RELU(in_channels_list[0], out_channels, stride = 1)
        self.output2 = Conv_BN1x1_RELU(in_channels_list[1], out_channels, stride = 1)
        self.output3 = Conv_BN1x1_RELU(in_channels_list[2], out_channels, stride = 1)

        self.merge1 = Conv_BN_RELU(out_channels, out_channels)
        self.merge2 = Conv_BN_RELU(out_channels, out_channels)

        # defining our quantizer and dequantizer
        # to be used in our forward pass 

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, input):
        # names = list(input.keys())
        # input = self.quant(input)

        # print(f'FPN: type of input:  {type(input)}  ')
        # print(f'FPN: input.shape:  {input.shape}  ')
        
        input = list(input.values())

        # input = self.quant(input)
        # print(f' input length: {len(input)}  ')
        # print(f' input [0]: {input[0]}  ')

        output1 = self.output1(self.quant(input[0]))
        output2 = self.output2(self.quant(input[1]))
        output3 = self.output3(self.quant(input[2]))

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [self.dequant(output1), self.dequant(output2), self.dequant(output3)]
        # dequantize output prior to returning it
        # out = self.dequant(out)
        return out
    
    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == Conv_BN1x1_RELU or type(m) == Conv_BN_RELU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            Conv_BN_RELU(3, 8, 2),    # 3
            Conv_DW(8, 16, 1),   # 7
            Conv_DW(16, 32, 2),  # 11
            Conv_DW(32, 32, 1),  # 19
            Conv_DW(32, 64, 2),  # 27
            Conv_DW(64, 64, 1)  # 43
        )
        self.stage2 = nn.Sequential(
            Conv_DW(64, 128, 2),  # 43 + 16 = 59
            Conv_DW(128, 128, 1), # 59 + 32 = 91
            Conv_DW(128, 128, 1), # 91 + 32 = 123
            Conv_DW(128, 128, 1), # 123 + 32 = 155
            Conv_DW(128, 128, 1), # 155 + 32 = 187
            Conv_DW(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            Conv_DW(128, 256, 2), # 219 +3 2 = 241
            Conv_DW(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

        # adding our quantizer and dequantizer 
        # for used in forward pass
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # quantize the input
        x = self.quant(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)

        # dequantize
        x = self.dequant(x)
        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == Conv_BN_RELU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == Conv_DW:
                # print(f'module: {m}')
                childs = [name for name,j in m.named_children()]
                # print(f'child modules: {childs}')
                for i in range(0,6,3):
                    # print(f'childs: {i} {childs[i:i+3]}')
                    torch.quantization.fuse_modules(m, childs[i+0:i+3], inplace=True)

# in the name of God the most compassionate the most mericiful 

import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH
from torchvision.models import quantization as quantized_models 
from torch.quantization import QuantStub, DeQuantStub 
from torch.nn.quantized import FloatFunctional 

from collections import OrderedDict

import torch
from torch import nn
from torch.jit.annotations import Dict
from torch.quantization import QuantStub, DeQuantStub 
from models.net import Conv_BN,Conv_BN1x1_RELU,Conv_DW,Conv_BN_RELU

class IntermediateLayerGetter(nn.ModuleDict):
    
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers

        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        layers = OrderedDict()
        # layers = nn.ModuleDict()
        
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
            
        # print(f'layers: {layers}')        
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        out = OrderedDict()
        # out = nn.ModuleDict()
        for name, module in self.items():
            x = self.quant(x)
            x = module(x)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                x = self.dequant(x)
                out[out_name] = x
        return out

    def fuse_model(self):
        # print('backend:(body): modules before quantization:')
        # print(list(self.children()))    
        for m in self.modules():
            if type(m) == Conv_BN_RELU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == Conv_DW:
                # print(f'module: {m}')
                childs = [name for name,j in m.named_children()]
                # print(f'child modules: {childs}')
                for i in range(0,6,3):
                    # print(f'childs: {i} {childs[i:i+3]}')
                    torch.quantization.fuse_modules(m, childs[i+0:i+3], inplace=True)
        # print('backend:(body): modules after quantization:')
        # print(list(self.children()))

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        # quantizer and dequantizer attributes
        # which will be used in forward pass
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models.quantization as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        # for mobilenet the net needs to be altered and then returned, so basically
        # the new resulting model must be quantized! 

        # the same thing goes to resnet, so we need to comeup with something that 
        # can be used with both of them and possible others 
        # print(f'mbnet before cut: {backbone}' )
        # we cant use ordered_dict

        # self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        self.body = IntermediateLayerGetter(backbone, cfg['return_layers'])
        # self.bodyq = nn.Sequential(QuantStub(),
        #                             backbone.stage1,
        #                             backbone.stage2,
        #                             backbone.stage3,
        #                             DeQuantStub())

        # self.bodyq.fuse_model = backbone.fuse_model
        # self.body = self.bodyq

        # print(f'mbnet after cut: {self.body}' )
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        #quantize
        inputs = self.quant(inputs)
        out = self.body(inputs)
        # FPN
        fpn = self.fpn(out)
        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]


        # should we use  torch.nn.quantized.FloatFunctional here? im not sure!!
        # bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        # classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        # ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        f_cat = FloatFunctional()
        bbox_regressions = f_cat.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = f_cat.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = f_cat.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        # dequantize 
        # is this even allowed? make sure to get back to this when things get weird!
        bbox_regressions = self.dequant(bbox_regressions)
        classifications = self.dequant(classifications)
        ldm_regressions = self.dequant(ldm_regressions)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        # dequantize
        # output = self.dequant(output)
        return output

    def fuse_model(self):
        #backends have their own fuse
        self.body.fuse_model()
        self.fpn.fuse_model()
        self.ssh1.fuse_model()
        self.ssh2.fuse_model()
        self.ssh3.fuse_model()

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}