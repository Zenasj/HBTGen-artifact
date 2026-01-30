import torch
import torch.nn as nn
import torchvision

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        
        self.mobile = models.mobilenet_v2(pretrained=True)
        in_features = self.mobile.classifier[-1].in_features
        out_features = 100
        self.mobile.classifier[-1] = nn.Linear(in_features=in_features, out_features=out_features)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.mobile(x)
        return self.softmax(x)

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = torchvision.datasets.CIFAR100(root=path, 
                                              train=True, 
                                              download=True, 
                                              transform=train_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           shuffle=True, 
                                           batch_size=4)

model = MobileNet()
optimzer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(reduction='sum')

i, (image, labels) = next(enumerate(train_loader))

out = model(image)

loss = criterion(out, labels)
loss.backward()