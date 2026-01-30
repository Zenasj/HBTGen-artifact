import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import csv

from torch.autograd import Variable

closeSet=[];

with open("data.csv",'r') as csvfile:
    reader=csv.reader(csvfile)

    n=-1;
    for row in reader :
        
        n=n+1;
        
        if(n == 0):
            continue;
        
        closeSet.append(float(row[6]));

closeSet=torch.Tensor(closeSet);

average=closeSet.mean();
print(average);

still=closeSet.std();
print(still);

closeSet=(closeSet-average)/still;
print(closeSet)

train_data=closeSet[:150];

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layer,output_size):
        super(LSTM, self).__init__()
        
        self.input_size=input_size;
        self.hidden_size=hidden_size;
        self.num_layer=num_layer;
        self.output_size=output_size;
        
        self.Wfh,self.Wfx,self.bf=self.init_Weight_bias_gate();
        self.Wih,self.Wix,self.bi=self.init_Weight_bias_gate();
        self.Woh,self.Wox,self.bo=self.init_Weight_bias_gate();
        self.Wch,self.Wcx,self.bc=self.init_Weight_bias_gate();
        self.Wy=nn.Parameter(torch.randn(self.hidden_size,self.output_size,requires_grad=True));
        self.by=nn.Parameter(torch.randn(self.output_size,requires_grad=True));
        
        self.hList=[];
        self.cList=[];

        self.times=0;
        
        self.f_=torch.zeros(1,self.hidden_size,requires_grad=True);
        self.i_=torch.zeros(1,self.hidden_size,requires_grad=True);
        self.o_=torch.zeros(1,self.hidden_size,requires_grad=True);
        self.ct_=torch.zeros(1,self.hidden_size,requires_grad=True);
        self.h_=torch.zeros(1,self.hidden_size,requires_grad=True);
        self.c_=torch.zeros(1,self.hidden_size,requires_grad=True);
        
        self.y_=torch.zeros(1,self.hidden_size,requires_grad=True);

        self.hList.append(self.h_);
        self.cList.append(self.c_);

    def init_Weight_bias_gate(self):
        return (nn.Parameter(torch.randn(self.hidden_size,self.hidden_size),requires_grad=True),
                nn.Parameter(torch.randn(self.input_size,self.hidden_size),requires_grad=True),
                nn.Parameter(torch.randn(self.hidden_size),requires_grad=True))
                
    def forward(self,x):
        self.times+=1;

        self.f_=torch.sigmoid(self.hList[-1] @ self.Wfh + x @ self.Wfx + self.bf);
        
        self.i_=torch.sigmoid(self.hList[-1] @ self.Wih + x @ self.Wix + self.bi);
        
        self.ct_=torch.tanh(self.hList[-1] @ self.Wch + x @ self.Wcx + self.bc);

        self.o_=torch.sigmoid(self.hList[-1] @ self.Woh + x @ self.Wox + self.bo);
        
        self.c_=self.f_ * self.cList[-1] + self.i_ * self.ct_;
        
        self.h_=self.o_ * torch.tanh(self.c_);
        self.y_=self.hList[-1] @ self.Wy + self.by;
        
        self.f_.requires_grad_(True);
        self.i_.requires_grad_(True);
        self.ct_.requires_grad_(True);
        self.o_.requires_grad_(True);
        self.c_.requires_grad_(True);
        self.h_.requires_grad_(True);
        
        self.y_.requires_grad_(True);
        
        self.cList.append(self.c_);
        self.hList.append(self.h_);
        
        return self.y_;
    
    def reset(self):
        self.times=0;
        
        self.hList=[torch.zeros(1,self.hidden_size,requires_grad=True),];
        self.cList=[torch.zeros(1,self.hidden_size,requires_grad=True),];

l=LSTM(150,1,1,150);

criterion = nn.MSELoss();

optimizer = torch.optim.Adam(l.parameters(), lr=0.01)

for i in range(10):
    x=train_data;
    x=x.clone().detach().unsqueeze(0);
    x.requires_grad_(True)
    
    y_output=l.forward(x);
    
    y_true=train_data;
    y_true=y_true.clone().detach();
    y_true.requires_grad_(True)
    
    loss=criterion(l.forward(x), y_true);
    
    optimizer.zero_grad();
    
    loss.backward();
    optimizer.step();
    
    print("Wfh.grad:");
    print(l.Wfh.grad)
    
    l.reset();

    lossList.append(loss.item())