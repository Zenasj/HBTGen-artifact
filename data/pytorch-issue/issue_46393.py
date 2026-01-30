import torchvision
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import datasets, transforms
import torch.nn.functional as F
import timeit
import unittest


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# check availability of GPU and set the device accordingly

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define a transforms for preparing the dataset
transform = transforms.Compose([
        transforms.ToTensor(), # convert the image to a pytorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalise the images with mean and std of the dataset
        ])

# Load the MNIST training, test datasets using `torchvision.datasets.MNIST` using the transform defined above

train_dataset = datasets.MNIST('./data',train=True,transform=transform,download=True)
test_dataset =  datasets.MNIST('./data',train=False,transform=transform,download=True)


# create dataloaders for training and test datasets
# use a batch size of 32 and set shuffle=True for the training set

train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)


# My Net

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # define a conv layer with output channels as 16, kernel size of 3 and stride of 1
        self.conv11 = nn.Conv2d(1, 16, 3, 1) # Input = 1x28x28  Output = 16x26x26
        self.conv12 = nn.Conv2d(1, 16, 5, 1) # Input = 1x28x28  Output = 16x24x24
        self.conv13 = nn.Conv2d(1, 16, 7, 1) # Input = 1x28x28  Output = 16x22x22

        # define a conv layer with output channels as 32, kernel size of 3 and stride of 1
        self.conv21 = nn.Conv2d(16, 32, 3, 1) # Input = 16x26x26 Output = 32x24x24
        self.conv22 = nn.Conv2d(16, 32, 5, 1) # Input = 16x24x24 Output = 32x20x20
        self.conv23 = nn.Conv2d(16, 32, 7, 1) # Input = 16x22x22 Output = 32x16x16

        # define a conv layer with output channels as 64, kernel size of 3 and stride of 1
        self.conv31 = nn.Conv2d(32, 64, 3, 1) # Input = 32x24x24 Output = 64x22x22
        self.conv32 = nn.Conv2d(32, 64, 5, 1) # Input = 32x20x20 Output = 64x16x16
        self.conv33 = nn.Conv2d(32, 64, 7, 1) # Input = 32x16x16 Output = 64x10x10

        # define a max pooling layer with kernel size 2
        self.maxpool = nn.MaxPool2d(2), # Output = 64x11x11
        # define dropout layer with a probability of 0.25
        self.dropout1 = nn.Dropout(0.25)
        # define dropout layer with a probability of 0.5
        self.dropout2 = nn.Dropout(0.5)
        # define a linear(dense) layer with 128 output features
        self.fc11 = nn.Linear(64*11*11, 128)
        self.fc12 = nn.Linear(64*8*8, 128)      # after maxpooling 2x2
        self.fc13 = nn.Linear(64*5*5, 128)

        # define a linear(dense) layer with output features corresponding to the number of classes in the dataset
        self.fc21 = nn.Linear(128, 10)
        self.fc22 = nn.Linear(128, 10)
        self.fc23 = nn.Linear(128, 10)

        self.fc33 = nn.Linear(30,10)
        

    def forward(self, x1):
        # Use the layers defined above in a sequential way (folow the same as the layer definitions above) and 
        # write the forward pass, after each of conv1, conv2, conv3 and fc1 use a relu activation. 
        

        x = F.relu(self.conv11(x1))
        x = F.relu(self.conv21(x))
        x = F.relu(self.maxpool(self.conv31(x)))
        #x = torch.flatten(x, 1)
        x = x.view(-1,64*11*11)
        x = self.dropout1(x)
        x = F.relu(self.fc11(x))
        x = self.dropout2(x)
        x = self.fc21(x)

        y = F.relu(self.conv12(x1))
        y = F.relu(self.conv22(y))
        y = F.relu(self.maxpool(self.conv32(y)))
        #x = torch.flatten(x, 1)
        y = y.view(-1,64*8*8)
        y = self.dropout1(y)
        y = F.relu(self.fc12(y))
        y = self.dropout2(y)
        y = self.fc22(y)

        z = F.relu(self.conv13(x1))
        z = F.relu(self.conv23(z))
        z = F.relu(self.maxpool(self.conv33(z)))
        #x = torch.flatten(x, 1)
        z = z.view(-1,64*5*5)
        z = self.dropout1(z)
        z = F.relu(self.fc13(z))
        z = self.dropout2(z)
        z = self.fc23(z)

        out = self.fc33(torch.cat((x, y, z), 0))
        
        output = F.log_softmax(out, dim=1)
        return output

import unittest

class TestImplementations(unittest.TestCase):
    
    # Dataloading tests
    def test_dataset(self):
        self.dataset_classes = ['0 - zero',
                                '1 - one',
                                '2 - two',
                                '3 - three',
                                '4 - four',
                                '5 - five',
                                '6 - six',
                                '7 - seven',
                                '8 - eight',
                                '9 - nine']
        self.assertTrue(train_dataset.classes == self.dataset_classes)
        self.assertTrue(train_dataset.train == True)
    
    def test_dataloader(self):        
        self.assertTrue(train_dataloader.batch_size == 32)
        self.assertTrue(test_dataloader.batch_size == 32)      
         
    def test_total_parameters(self):
        model = Net().to(device)
        #self.assertTrue(sum(p.numel() for p in model.parameters()) == 1015946)

suite = unittest.TestLoader().loadTestsFromModule(TestImplementations())
unittest.TextTestRunner().run(suite)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # send the image, target to the device
        data, target = data.to(device), target.to(device)
        # flush out the gradients stored in optimizer
        optimizer.zero_grad()
        # pass the image to the model and assign the output to variable named output
        output = model(data)
        # calculate the loss (use nll_loss in pytorch)
        loss = F.nll_loss(output, target)
        # do a backward pass
        loss.backward()
        # update the weights
        optimizer.step()
      
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
          
            # send the image, target to the device
            data, target = data.to(device), target.to(device)
            # pass the image to the model and assign the output to variable named output
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
          
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

model = Net().to(device)

## Define Adam Optimiser with a learning rate of 0.01
optimizer =  torch.optim.Adam(model.parameters(),lr=0.01)

start = timeit.default_timer()
for epoch in range(1, 11):
  train(model, device, train_dataloader, optimizer, epoch)
  test(model, device, test_dataloader)
stop = timeit.default_timer()
print('Total time taken: {} seconds'.format(int(stop - start)) )