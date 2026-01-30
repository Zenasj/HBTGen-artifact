import torch.nn as nn

input_size = 784 # 28 * 28
hidden_size = 500
num_classes = 10

class NeuralNet(nn.Module):
    def __init__(self , input_size , hidden_size , num_classes):
        super(NeuralNet , self).__init__()
        self.l1 = nn.Linear(input_size , hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size , num_classes )
        self.softmax  = nn.Softmax()
    def forward(self , x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.softmax(out)
        return out

model = NeuralNet(input_size , hidden_size , num_classes).to(device)  # --> error generates here

CUDA_LAUNCH_BLOCKING = 1