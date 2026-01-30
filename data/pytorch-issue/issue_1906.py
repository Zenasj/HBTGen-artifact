import torch
import torch.nn as nn                                                                                                                                                                                              
from torch.autograd import Variable                                                                                                                                                                                
                                                                                                                                                                                                                    
net = nn.MaxPool2d(2)#.cuda()                                                                                                                                                                                      
#net = nn.AvgPool2d(2)#.cuda()                                                                                                                                                                                      
                                                                                                                                                                                                                    
xd = torch.FloatTensor(1, 3, 448, 448)                                                                                                                                                                           
x = Variable(xd, volatile=True)#.cuda()                                      `                                                                                                                                      
y = net.forward(x)                                                                                                                                                                                                 
print(y.data.mean())