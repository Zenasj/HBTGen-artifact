import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1_0 = self.Conv_Module(3, 6, 1, 3, 1)
        self.layer1_1 = self.Conv_Module(1, 6, 4, 3, 1)
        self.Dense = self.Classification(384)

    def Conv_Module(self, in_filters, out_filters, stride, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_filters, out_channels=out_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU())

    def Classification(self, num_in):
        return nn.Sequential(
            nn.Linear(num_in, 10))

    def forward(self, original_img):
        first_conv = self.layer1_0(original_img)
        attention_map = first_conv.mean(1).unsqueeze(dim = 1)
        inverted_attention = -attention_map
        output_mask = torch.ge(attention_map, inverted_attention).float()
        second_conv = self.layer1_1(output_mask)
        Classifier = self.Dense(second_conv.view(-1, 384))
        return F.log_softmax(Classifier, dim=1)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                           shuffle=True, num_workers=1)

model = Net()
optimizer = optim.Adam(model.parameters(), lr=1.5)
model.train()

for batch_idx, (data, target) in enumerate(train_loader):
    model.train()
    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    print("Weights before .ge:" , torch.sum(model.layer1_0[0].weight))
    print("Weights after .ge:" ,torch.sum(model.layer1_1[0].weight))
    print("*********")