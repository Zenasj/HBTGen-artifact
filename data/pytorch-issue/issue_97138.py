import torch.nn as nn

import torch.nn.functional as F
from torch import nn
from torchsummary import summary
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")
print(device)
print(f"Using {device} device")
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        constant1 = torch.tensor([
            -0.1615397185087204,
            -0.4338356554508209,
            0.09164135903120041,
            -0.01685221679508686,
            -0.06502643972635269,
            -0.1317378729581833,
            0.020417550578713417,
            -0.1211102306842804
            ])
        constant2 = torch.tensor([
            -0.08224882185459137,
            -0.10886877775192261,
            -0.14103959500789642,
            -0.20486916601657867,
            -0.17913565039634705,
            -0.2154383808374405,
            -0.1338050663471222,
            -0.19572456181049347,
            -0.26825064420700073,
            -0.25821220874786377,
            -0.07615606486797333,
            0.01328414585441351,
            -0.004444644320756197,
            -0.41474083065986633,
            -0.17879115045070648,
            -0.03865588828921318])
        constant3 = torch.randn(16,4,4,10)
        constant4 = torch.tensor([[
                -0.04485602676868439,
                0.007791661191731691,
                0.06810081750154495,
                0.02999374084174633,
                -0.1264096349477768,
                0.14021874964237213,
                -0.055284902453422546,
                -0.04938381537795067,
                0.08432205021381378,
                -0.05454041436314583
            ]])
        self.constant1 = nn.Parameter(data = constant1)
        self.constant2 = nn.Parameter(data = constant2)
        self.constant3 = nn.Parameter(data = constant3)
        self.constant4 = nn.Parameter(data = constant4)
#         self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(5,5),stride=(1,1),padding='same',dilation=(1,1),groups=1)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(5,5),stride=(1,1),padding=2,dilation=(1,1),groups=1)
        self.relu1 = nn.ReLU()
        self.reshape1 = nn.Unflatten(0,(1,8,1,1))
        self.reshape2 = nn.Unflatten(0,(1,16,1,1))
        self.reshape3 = nn.Sequential(
                    nn.Flatten(start_dim=0),
                    nn.Unflatten(0,(256,10))
        )
        self.reshape4 = nn.Sequential(
                    nn.Flatten()
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
#         self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(5,5),stride=(1,1),padding='same',dilation=(1,1),groups=1)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(5,5),stride=(1,1),padding=2,dilation=(1,1),groups=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3),stride=(3,3))
    def forward(self, x):
        x = x/255
        x = self.conv1(x)
        reshape1_output = self.reshape1(self.constant1)
        x = x + reshape1_output
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        reshape2_output = self.reshape2(self.constant2)
        x = x + reshape2_output
        x = self.relu2(x)
        x = self.maxpool2(x)
#         print(self.constant3.shape)
        reshape3_output = self.reshape3(self.constant3)
#         print(reshape3_output.shape)
        x = self.reshape4(x)
#         print(x.shape)
        x = torch.mm(x,reshape3_output)
        x = x + self.constant4
        return x
#
print(device)
model = NeuralNetwork().to(device)
# input = torch.randn(1,28,28)
summary(model=model,input_data=(1,1,28,28),batch_dim = None,device=device)

dummy_input = torch.randn(1, 1, 28, 28, device='cuda:1')
input_names = [ "input73" ] 
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "mnist1.onnx", verbose=True, input_names=input_names, output_names=output_names)