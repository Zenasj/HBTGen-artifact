import torch

model = UNet(n_channels=1,
             mode='3D',
             num_classes=1,
             use_pooling=True,
             )

def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_steps_train = len(train_loader)
    
    print(num_steps_train)
    
    for epoch in range(epochs):
        print(' - training - ')
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model.forward(images)

train_loader = DataLoader(dataset=Dataset(partition['orig'], partition['segment']), 
                          batch_size = batch_size, shuffle = True)