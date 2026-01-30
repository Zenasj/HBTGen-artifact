KeyError: 'masked_select'

KeyError: 'nonzero_numpy'

import torch
# import onnx
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F

class mymodel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,):

        super().__init__()
        self.name = 'PFNLayer'
        # self.units = out_channels
        # self.batchnorm1d=nn.BatchNorm1d(eps=1e-3,momentum=0.01)
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels,eps=1e-3,momentum=0.01)

    def forward(self, inputs):
        if inputs.shape[0]>65000:
            inputs1,inputs2,inputs3=torch.chunk(inputs,3,dim=0)
            x1=self.linear(inputs1)
            x2 = self.linear(inputs2)
            x3 = self.linear(inputs3)
            x=torch.cat([x1,x2,x3],dim=0)
        else:
            x=self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        x_concatenated = torch.cat([x, x_repeat], dim=2)
        # masked=x_concatenated>5
        idx=torch.where(x_concatenated>5)[0]
        # idx=torch.nonzero(x_concatenated>5,as_tuple=True)[0]
        x_concatenated=x_concatenated[idx]
        # x_concatenated=x_concatenated.masked_select(masked)
        return x_concatenated



if __name__=="__main__":
    inputs=torch.randn((5000,100,30),dtype=torch.float)
    net=mymodel(30,9)
    torch.onnx.export(net,(inputs,),'onnx_test',verbose=True)