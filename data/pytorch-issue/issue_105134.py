import torch
import torch.nn as nn
import torch.nn.functional as F

x=torch.randn(176,64,56,56)
y=torch.randn(176,128,28,28)

input_names=['input']
output_names=['output']

class embedding_concat(nn.Module):
    def __init__(self):
        super(embedding_concat,self).__init__()
        
    def forward(self,x):
        
        y=torch.randn(176,128,28,28)
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        print(z.shape)
        fold=nn.Fold(output_size=(H1,W1),kernel_size=s,stride=s)
        z = fold(z)
        
        return z



model=embedding_concat()
torch.onnx.export(model,
                  x,
                  "embedding_concat.onnx",
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=18, 
                  do_constant_folding=False, # 是否压缩常量
                  #export_params=False,
                  dynamic_axes={"input":{1: "channel",2:"h",3:"w"},}
                  )