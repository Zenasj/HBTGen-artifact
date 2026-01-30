import torch
import torch.nn as nn
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.layer1=nn.Conv3d(1,64,5,padding=2) 
        
        self.layer2 = nn.Conv3d(64,1,1)
        self.sigmoid=nn.Sigmoid()
            
            
    def forward(self,in_vol):
        
        first = self.layer1(in_vol)

        MonoClass=self.layer2(first)
        
        Mask=self.sigmoid(MonoClass)
        
        return Mask


A=Net().cuda()
batch=1

SideSize=256
Zsize=4
X=torch.randn(batch,1,SideSize,SideSize,Zsize)
X=X.cuda()

optimizer=torch.optim.Adam(A.parameters())

optimizer.zero_grad()
Mask = A(X)

loss=torch.sum(Mask)
loss=loss
loss.backward()
optimizer.step()
print(loss)