import torch
import torch.nn as nn

def main():
    train_set = MockDataset()
    train_loader = DataLoader(train_set, batch_size=4, num_workers=1, drop_last=True)
    model = UNet(n_classes=13)
    print(model)
    device = 'cpu'
    model.to(device)
    optimizer = SGD(model.parameters(), lr=1e-3)
    for epoch in range(1):
        model.train()
        for step, batch_data in enumerate(train_loader):
            inputs = batch_data[0].to(device)
            labels = batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs.shape, labels.shape)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            print(f"\nBefore HANG {loss}\n")
            loss.backward()
            print("\nAFTER HANG\n")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()