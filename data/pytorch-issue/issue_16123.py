import torch.nn as nn

import torch
from torch import nn

class SubModule(torch.jit.ScriptModule):
    def __init__(self):
        super(SubModule, self).__init__()
        self.weight = nn.Parameter(torch.randn(2))

    @torch.jit.script_method
    def forward(self, input):
        return self.weight + input

class MyModule(torch.jit.ScriptModule):
    __constants__ = ['mods']

    def __init__(self):
        super(MyModule, self).__init__()
        self.mods = torch.nn.ModuleList([SubModule() for i in range(10)])

    @torch.jit.script_method
    def forward(self, v):
        for module in self.mods:
            v = m(v)
        return v
MyModule()

i = 0
for mod in self.modules:
   mod(output[i])
   i += 1

class MyMod(torch.jit.ScriptModule):
    __constants__ = ['loc_layers']
    def __init__(self):
        super(MyMod, self).__init__()
        locs = []
        for i in range(4):
            locs.append(nn.Conv2d(64, 21, kernel_size=1))
        self.loc_layers = nn.ModuleList(locs)

    @torch.jit.script_method
    def forward(self, input):
        # type: (List[Tensor]) -> Tensor
        locs = []
        i = 0
        for layer in input:
            loc = self.loc_layers[i](layer)
            locs.append(loc)
            i += 1
        loc_preds = torch.cat(locs, 1)
        return loc_preds

i = 0
for layer in self.loc_layers:
    loc = layer(input[i])
    locs.append(loc)
    i += 1

3
class VGG16_frontend(nn.Module):
    def __init__(self,block_num=5,decode_num=0,load_weights=True,bn=False,IF_freeze_bn=False):
        super(VGG16_frontend,self).__init__()
        self.block_num = block_num
        self.load_weights = load_weights
        self.bn = bn
        self.IF_freeze_bn = IF_freeze_bn
        self.decode_num = decode_num

        block_dict = [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'],\
             [512, 512, 512,'M'], [512, 512, 512,'M']]

        self.frontend_feat = []
        for i in range(block_num):
            self.frontend_feat += block_dict[i]

        if self.bn:
            self.features = make_layers(self.frontend_feat, batch_norm=True)
        else:
            self.features = make_layers(self.frontend_feat, batch_norm=False)


        if self.load_weights:
            if self.bn:
                pretrained_model = models.vgg16_bn(pretrained = True)
            else:
                pretrained_model = models.vgg16(pretrained = True)
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # load the new state dict
            self.load_state_dict(model_dict)

        if IF_freeze_bn:
            self.freeze_bn()
    
    def forward(self,x):
        if self.bn: 
            x = self.features[ 0:7](x)
            print(type(x))
            print(x.shape)
            conv1_feat =x if self.decode_num>=4 else []
            x = self.features[ 7:14](x)
            conv2_feat =x if self.decode_num>=3 else []
            x = self.features[ 14:24](x)
            conv3_feat =x if self.decode_num>=2 else []
            x = self.features[ 24:34](x)
            conv4_feat =x if self.decode_num>=1 else []
            x = self.features[ 34:44](x)
            conv5_feat =x 
        else:
            x = self.features[ 0: 5](x)
            conv1_feat =x if self.decode_num>=4 else []
            x = self.features[ 5:10](x)
            conv2_feat =x if self.decode_num>=3 else []
            x = self.features[ 10:17](x)
            conv3_feat =x if self.decode_num>=2 else []
            x = self.features[ 17:24](x)
            conv4_feat =x if self.decode_num>=1 else []
            x = self.features[ 24:31](x)
            conv5_feat =x 
               
        feature_map = {'conv1':conv1_feat,'conv2': conv2_feat,\
            'conv3':conv3_feat,'conv4': conv4_feat, 'conv5': conv5_feat}   
        
        # feature_map = [conv1_feat, conv2_feat, conv3_feat, conv4_feat, conv5_feat]
        
        return feature_map


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()