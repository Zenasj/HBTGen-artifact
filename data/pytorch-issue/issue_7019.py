import torch
import torch.nn as nn
from torch.autograd import Variable

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.dense = nn.Linear(256, 512)

    def forward(self, input):
        return self.dense(input)

if __name__ == '__main__':
    model = NET()
    model = nn.DataParallel(model).cuda()
    x = Variable(torch.rand(128, 256))
    y = model(x) ##### <<<<--- GETS STUCK HERE FOREVER