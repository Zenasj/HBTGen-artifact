import torch
import torch.nn as nn
import torch.optim as optim
from make_data import train_dataloader, test_dataloader
from make_net import net, Net
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os


# os.environ["CUDA_VISIBLE_DEVICES"] ="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()
opt_Adam = optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.99))
scheduler = ReduceLROnPlateau(opt_Adam, mode='min')

def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    train_loader =train_dataloader
    net = model()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net)
    net.to(device)

    criterion = criterion
    optimizer = optimizer
    scheduler = scheduler

    for epoch in range(num_epochs):
        running_loss = 0.0
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-" * 10)

        for i, sample in enumerate(train_loader, 0):
            image, pressure = sample['image'], sample['pressure']

            image = image.float()
            image = image.to(device)

            pressure = pressure.float()
            pressure = pressure.to(device)

            optimizer.zero_grad()
            output = net(pressure)
            loss = criterion(output, image)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print("%d, %5d, loss: %.3f" % (epoch, i, running_loss/100))
                running_loss = 0.0
        # scheduler.step()


train_model(model=Net, criterion=criterion, optimizer=opt_Adam, scheduler=scheduler)