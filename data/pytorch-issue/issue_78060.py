import torch.nn as nn

import torch
import torch.onnx
from torch import nn

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        # self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.conv3 = nn.MultiheadAttention(512,8,dropout=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # out = self.relu(self.conv1(x))
        # out = self.relu(self.conv2(out))
        # out = self.conv3(out)
        out = torch.randn([200,1, 512])
        out2 = torch.randn([200, 1, 512])
        out3 = torch.randn([200, 1, 512])

        # the value of  key and value must be the same,or there will be an error
        out = self.conv3(query=out, key=out, value=out2)[0]

        # print(out)
        return out


def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)
    state_dict = torch.load('srcnn.pth')['state_dict']
    state_dict['conv3.in_proj_weight']=torch.randn((1536,512))
    state_dict['conv3.in_proj_bias']=torch.randn(1536)
    state_dict['conv3.out_proj.weight'] = torch.randn((512,512))
    state_dict['conv3.out_proj.bias']=torch.randn(512)
    state_dict2={}
    for key in state_dict.keys():
        if key!='generator.conv3.weight' and key !='generator.conv3.bias':
            if 'generator' not in key:
                state_dict2[key]=state_dict[key]
            else:
                state_dict2[key.replace('generator.','')]=state_dict[key]


    torch_model.load_state_dict(state_dict2)
    torch_model.eval()
    return torch_model


model = init_torch_model()

x = torch.randn(1, 3, 256, 256)


with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "test.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'])

torch.backends.mha.set_fastpath_enabled(False)