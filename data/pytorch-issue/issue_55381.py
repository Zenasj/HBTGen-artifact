import torch
import torch.nn as nn

torch.manual_seed(0)

in_channels = 64
out_channels = 128
stride = 2
W = 32
H = 32

image = torch.rand(1, in_channels, W, H)
net = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
torch.save(net, '.\model.pkl')

def runcpu(image):
    net = torch.load('.\model.pkl')
    net.eval()
    with torch.no_grad():        
        output = net(image)
    return output

def runCuDNN(image):
    net = torch.load('.\model.pkl').cuda()
    net.eval()
    with torch.no_grad():        
        torch.backends.cudnn.enabled = True
        output = net(image.cuda())
    return output

def runNoCuDNN(image):
    net = torch.load('.\model.pkl').cuda()
    net.eval()
    with torch.no_grad():        
        torch.backends.cudnn.enabled = False
        output = net(image.cuda())
    return output


noCuDNN_output = runNoCuDNN(image)
cpu_output = runcpu(image)
CuDNN_output = runCuDNN(image)
print(cpu_output.sum())
print(CuDNN_output.sum())
print(noCuDNN_output.sum())

tensor(502.4232)
tensor(502.4232, device='cuda:0')
tensor(541.0007, device='cuda:0')