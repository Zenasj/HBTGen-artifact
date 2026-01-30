import torch.nn as nn
import torch 
import argparse
import torch,time

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size,bias=False)  # 1st Full-Connected Layer:  (input data) ->  (hidden node)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, output_size,bias=False) # 2nd Full-Connected Layer:  (hidden node) ->  (output class)        
        nn.init.kaiming_normal_(self.fc2.weight)
        self.LRelu = nn.LeakyReLU(0.25)
        self.fnn = nn.Sequential(
            self.fc1,
            self.LRelu,
            self.fc2
        ) 

def forward(self,x):
        return self.fnn(x)
           
device= torch.device('cuda')
start = time.time()
nodes_num=100
x = torch.ones(nodes_num, 12).to(device)
y = torch.ones(nodes_num, 1).to(device)
dummy = torch.zeros(12*nodes_num,nodes_num)
dummy[0][0]=1
dummy=dummy.to_sparse()
dummy.to(device)
print(dummy)
parent=torch.ones(12*nodes_num,1).to (device)
net = Net(12,4,1)
net.to(device)
optimizer = torch.optim.RMSprop(net.parameters())
criterion = nn.MSELoss()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
res=net(x)
new_parent=torch.sparse.mm(dummy,res)
end_event.record()
torch.cuda.synchronize()  # Wait for the events to be recorded!
elapsed_time_ms = start_event.elapsed_time(end_event)