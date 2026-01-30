def test_torch():
    import torch
    import torch.nn as nn
    from torch.utils import data
    from torch.utils.data import dataloader

    data = torch.randn(100000, 2)
    y = torch.sin(data[:, 0]*data[:, 1]).unsqueeze(1)
    print(y.shape)
    data = torch.cat([data, y], dim=1)

    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 10), 
        nn.ReLU(),
        nn.Linear(10, 10), 
        nn.ReLU(),
        nn.Linear(10, 10), 
        nn.ReLU(),
        nn.Linear(10, 10), 
        nn.ReLU(),
        nn.Linear(10, 10), 
        nn.ReLU(),
        nn.Linear(10, 10), 
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    for p in model.parameters():
        print(p.shape, p.requires_grad)

    criterion = torch.nn.L1Loss()
    opt = torch.optim.SGD(model.parameters(), 0.001)

    dl = dataloader.DataLoader(data, batch_size=500)
    for epoch in range(100):
        for d in dl:
            pred = model(d[:, :2])
            loss = criterion(pred, d[:, 2])
            opt.zero_grad()
            loss.backward()
            opt.step()
            print("epoch:[{}], loss: {}".format(epoch, loss.item()))
if __name__ == "__main__":
    test_torch()