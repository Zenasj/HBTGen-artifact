import torch.nn as nn
import torch.nn.functional as F
import random

import torch
import numpy as np
import time
import os
from torchvision import transforms
from torch import nn
#from resnet import test_resnet

INPUT_SHAPE = {512:(512,960), 704:(704,1280), 1056:(1056,1920)}

data_transforms = transforms.Compose(
	[
	#transforms.Resize(INPUT_SHAPE ),
	#transforms.CenterCrop(INPUT_SHAPE ),            
	transforms.ToTensor(),
	transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
	]
)

if __name__=='__main__':
	torch.backends.cudnn.benchmark = True
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device('cuda')

	#model1 = test_resnet
	#model = model1

	model2 = torch.jit.load('traced_model.pt')
	model = model2
	model = model.to(device)
	model.eval()
	inputSize = INPUT_SHAPE[704]
	samples = []
	for i in range(10):
		samples.append( np.random.random((inputSize[0],inputSize[1],3)).astype(np.float32) )
	print('warming up')
	for i, sample in enumerate(samples):
		t0 = time.time()
		print(sample.shape)
		image = data_transforms(sample).unsqueeze(0).to(device)
		if(i==0 and False):
			traced_script_module = torch.jit.trace(model, image)
			traced_script_module.save("traced_model.pt")

		torch.cuda.synchronize()
		t1 = time.time()
		outputs = model(image)
		torch.cuda.synchronize()
		t2 = time.time()
		raw = outputs.cpu().detach().numpy()
		torch.cuda.synchronize()
		t3 = time.time()
		print('warm step({}) time: {} {} {} {}'.format(i, t1-t0,t2-t1,t3-t2, t3-t0))
	ct1 = 0.
	ct2 = 0.
	ct3 = 0.
	ct4 = 0.
	for k in range(20):
		samples = []
		for i in range(10):
			samples.append( np.random.random((inputSize[0],inputSize[1],3)).astype(np.float32) )
		for i, sample in enumerate(samples):
			t0 = time.time()
			image = data_transforms(sample).unsqueeze(0).to(device)
			torch.cuda.synchronize()
			t1 = time.time()
			outputs = model(image)
			torch.cuda.synchronize()
			t2 = time.time()
			raw = outputs.cpu().detach().numpy()
			torch.cuda.synchronize()
			t3 = time.time()
			ct1 += t1-t0
			ct2 += t2-t1
			ct3 += t3-t2
			ct4 += t3-t0
		ct1 /= 10
		ct2 /= 10
		ct3 /= 10
		ct4 /= 10
		print('test batch({}) time: pr:{} m:{} po:{} t:{}'.format(k, ct1,ct2,ct3, ct4))
		ct1 = 0.
		ct2 = 0.
		ct3 = 0.
		ct4 = 0.

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)


        self.conv_top_512_128 = conv1x1(512, 128)
        self.bn1_ = nn.BatchNorm2d(128)
        self.act1 = nn.ReLU(inplace=True)

        self.conv_cat_384 = conv3x3(384, 256)
        self.bn2 = nn.BatchNorm2d(256)
        self.act2 = nn.ReLU(inplace=True)

        self.conv_cat_256 = conv3x3(256, 1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        x3 = self.conv_top_512_128(x3)
        x3 = self.bn1_(x3)
        x3 = self.act1(x3)
        c3 = F.interpolate(x3, scale_factor=2, mode='bilinear')

        c2 = torch.cat([c3,x2], dim = 1)
        c2 = self.conv_cat_384(c2)
        c2 = self.bn2(c2)
        c2 = self.act2(c2)

        x = self.conv_cat_256(c2)

        return torch.abs(x)

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress):
    model = ResNet(block, layers)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        backbone_state_dict = model.state_dict()
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in backbone_state_dict }
        #print(pretrained_dict)
        backbone_state_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        #model.load_state_dict(state_dict)
    return model

def resnet34(pretrained=False, progress=True):
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress)