import torch.nn as nn

import torch
from torch.export import Dim

class ConvNN(nn.Module):
    def __init__(self, params):

        super(ConvNN, self).__init__()
        self.num_output = 0
        self.num_classes = params['num_classes']
        self.confidence_threshold = params['confidence_threshold']
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3,64, kernel_size=3,padding=1, stride=1)
        self.batch_norm1 = nn.BatchNorm2d((64))

        self.conv2 = nn.Conv2d(64,64, kernel_size=3,padding=1, stride=1)
        self.batch_norm2 = nn.BatchNorm2d((64))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.early_exit1 = nn.Linear(16384, self.num_classes)

    def forward(self, x):
        x= self.conv1(x)
        x=self.batch_norm1(x)
        x= self.relu(x)

        x= self.conv2(x)
        x=self.batch_norm2(x)
        x= self.maxpool1(x)
        x= self.relu(x)

        return x, nn.functional.softmax(self.early_exit1(x.clone().view(x.size(0), -1)), dim=1)
    
###-- Main
def main():


    x = torch.rand(32,3,32,32).to('cuda')
    params = {}
    params['num_classes'] = 10
    params['confidence_threshold'] = 0.5
    model = ConvNN(params)
    model.cuda()
    model.eval()
    batch = Dim("batch")
    dynamic_shapes = {"x": {0: batch}}

    torch.export.export(model, (x,) , dynamic_shapes=dynamic_shapes)

if __name__ == '__main__':
    main()