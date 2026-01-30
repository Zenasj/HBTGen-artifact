import torch.nn as nn

import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/temp')
net = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

labels = torch.randint(10, size=(2,))
for i in range(5):
    # net.train()  <--- uncomment this line then no error
    print(i)
    optimizer.zero_grad()
    output = net(torch.randn(2, 3, 1, 1))
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    net.eval()  # <--- or comment this line then no error as well
    for name, w in net.named_parameters():
        writer.add_histogram(name, w, i)