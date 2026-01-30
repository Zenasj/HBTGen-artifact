import torch.nn as nn

class TestView(nn.Module):
    def __init__(self):
        super(TestView, self).__init__()
        self.conv2d = nn.Conv2d(22, 32, kernel_size=1, bias=True)
        self.fc = nn.Linear(15488, 3)

    def forward(self, x):
        x = self.conv2d(x)
        
        print(type(x.size()[0]))

        x = x.view([x.size(0), -1])
        x = self.fc(x)
        return x