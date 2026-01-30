import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True)


 
class fuseNet(nn.Module):
    def __init__(self):
        super(fuseNet, self).__init__()
        self.dropOut = nn.Dropout(p=0.2)
        self.convB1_BW = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.ReLU(),
        )
        
        self.convB2_BW = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
       
        self.convB3_BW = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.convB4_BW = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
       
        self.convB5_BW = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        
        
        
        
        
        self.convB1_RGB = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.convB2_RGB = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        
        self.convB3_RGB = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.convB4_RGB = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        
        self.convB5_RGB = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        
        
        
        
        
        
        
        self.convB1_FS = nn.Sequential(
            nn.Conv2d(320, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.convB2_FS = nn.Sequential(
            nn.Conv2d(640, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.convB3_FS = nn.Sequential(
            nn.Conv2d(1280, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.convB4_FS = nn.Sequential(
            nn.Conv2d(1536, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.convB5_FS = nn.Sequential(
            nn.Conv2d(2560, 1024, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        
        
        
        self.decoder = nn.Sequential(
            nn.Linear(1024 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 84),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(84, 2),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, img, cimg): 
        print(img.shape)
        img = img.unsqueeze(0)
        print(img.shape)
        x = self.convB1_BW(img)
        x_1 = x
        x = self.convB2_BW(x)
        x_2 = x
        x = self.dropOut(x)
        x = self.convB3_BW(x)
        x_3 = x
        x = self.convB4_BW(x)
        x_4 = x
        x = self.dropOut(x)
        x = self.convB5_BW(x)
        x_5 = x
        
        
        y = self.convB1_RGB(cimg)
        y_1 = y
        y = self.convB2_RGB(y)
        y_2 = y
        y = self.dropOut(y)
        y = self.convB3_RGB(y)
        y_3 = y
        y = self.convB4_RGB(y)
        y_4 = y
        x = self.dropOut(y)
        x = self.convB5_RGB(y)
        y_5 = y
        
        
        
        # stage 1
        z = self.convB1_FS(torch.cat((y_1, x_1), dim=1))
        temp_t = torch.add(x_1, y_1)
        z = torch.add(z, temp_t)
        
        # stage 2
        z = self.convB2_FS(torch.cat((y_2, x_2, z), dim=1))
        z = torch.add(z, torch.add(x_2, y_2))
        
        # stage 3
        z = self.convB3_FS(torch.cat((y_3, x_3, z), dim=1))
        z = torch.add(z, torch.add(x_3, y_3))
        
        # stage 4
        z = self.convB4_FS(torch.cat((y_4, x_4, z), dim=1))
        z = torch.add(z, torch.add(x_4, y_4))
        
        # stage 5
        z = self.convB5_FS(torch.cat((y_5, x_5, z), dim=1))
        z = torch.add(z, torch.add(x_5, y_5))
        
        # dropout before flattening
        z = self.dropOut(z)
        
        # flattening
        z = z.view(-1)
        
        # classiffier
        output = self.decoder(z)
        return output

fuseNet = fuseNet().to(device)
optimizer = torch.optim.Adam(fuseNet.parameters(), lr = 0.0001)
criterion = nn.BCELoss() # binary cross entropy


epochs = 15
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []


for e in range(epochs):
  
  running_loss = 0.0
  running_corrects = 0.0
  val_running_loss = 0.0
  val_running_corrects = 0.0
  
  for image, cimage, labels in train_loader:
    image = image.to(device)
    #image = image.unsqueeze(0)
    # print(image.shape)
    cimage = cimage.to(device)
    labels = labels.to(device)
    
    outputs = fuseNet(image, cimage)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    _, preds = torch.max(outputs, 1)
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data)
 
  else:
    with torch.no_grad():
      for val_inputs, val_labels in test_loader:
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        val_outputs = fuseNet(val_inputs)
        val_loss = criterion(val_outputs, val_labels)
        
        _, val_preds = torch.max(val_outputs, 1)
        val_running_loss += val_loss.item()
        val_running_corrects += torch.sum(val_preds == val_labels.data)
      
    epoch_loss = running_loss/len(train_loader)
    epoch_acc = running_corrects.float()/ len(train_loader)
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_acc)
    
    val_epoch_loss = val_running_loss/len(test_loader)
    val_epoch_acc = val_running_corrects.float()/ len(test_loader)
    val_running_loss_history.append(val_epoch_loss)
    val_running_corrects_history.append(val_epoch_acc)
    print('epoch :', (e+1))
    print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
    print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))