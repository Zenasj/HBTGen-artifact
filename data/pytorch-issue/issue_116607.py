import torch.nn as nn

import argparse
import time

import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size.",
)
parser.add_argument("--num_data", default=1024, type=int, help="Number of fake images.")
args = parser.parse_args()


train_dataset = datasets.FakeData(
    args.num_data, (3, 224, 224), 1000, transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True
)
test_dataset = datasets.FakeData(
    args.num_data, (3, 224, 224), 1000, transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True
)

model = torchvision.models.resnet50(pretrained=True)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


print("==================== Training ====================")
model.train()
for i, (images, target) in enumerate(train_loader):
    optimizer.zero_grad()

    start = time.time()
    outputs = model(images)
    end = time.time()
    print(f"Train forward time: {(end - start) * 1000.0} ms")

    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()


print("==================== Evaluation ====================")
model.eval()
for i, (images, target) in enumerate(test_loader):
    with torch.no_grad():
        start = time.time()
        outputs = model(images)
        end = time.time()
        print(f"Eval forward time: {(end - start) * 1000.0} ms")