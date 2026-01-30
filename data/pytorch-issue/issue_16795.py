import torch
import torch.nn as nn
import numpy as np
import random

class age_dataset(Dataset):
  def __init__(self,label,image_dir,transform = None):
    self.data = pd.read_csv(label)
    self.img_dir = image_dir
    self.transform = transform
    self.id = self.data.iloc[:,1:2].values
  def __len__(self):
        return len(self.id)
  def __getitem__(self,idx):
    img_name = os.path.join(self.img_dir,self.id[idx,0])
    image = cv2.imread(img_name)
    age1 = self.data.iloc[:,2:3].values

    age = age1[idx,:]
    if self.transform:
      image = self.transform(image)
    return image,int(age)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.39125082, 0.45353994, 0.5837345 ], [0.20129617, 0.21080811, 0.24163313])
    ])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.39125082, 0.45353994, 0.5837345 ], [0.20129617, 0.21080811, 0.24163313])
    ])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
age_data = age_dataset('./agedetect/label.csv','./UTKFace' , transform = train_transform)

test_size = 0.2
valid_size = 0.1

data_len = len(age_data)
indices = list(range(data_len))

np.random.shuffle(indices)

split1 = int(np.floor(valid_size * data_len))
split2 = int(np.floor(test_size * data_len))
split2 = split2+split1

valid_idx , test_idx, train_idx = indices[:split1], indices[split1:split2] , indices[split2:]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = DataLoader(age_data, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(age_data, batch_size=32 , sampler=valid_sampler)
test_loader = DataLoader(age_data, batch_size=32 , sampler=test_sampler)

dataloaders = {'train':train_loader,'val':valid_loader}

resnet50 = models.resnet18(pretrained=True)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(nn.Linear(num_ftrs,256),nn.ReLU(),nn.Linear(256,104),nn.LogSoftmax(dim=1))

resnet50 = resnet50.to(device)
criterion = nn.CrossEntropyLoss()


optimizer_ft = optim.Adam(resnet50.parameters(), lr=0.1)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    model.to(device)

    best_acc = 0.0
    i = 0
    for epoch in range(num_epochs):
      print('Epoch:',epoch)
      
      for phase in ['train', 'val']:
        if phase == ' train':
            scheduler.step()
            model.train()  
        else:
            model.eval()   
            
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for inputs, labels in dataloaders[phase]:    
            labels = labels.to(device)
            #labels = labels.type(torch.cuda.FloatTensor)
            inputs = inputs.view(inputs.shape[0],3,224,224)
            inputs = inputs.to(device)
            inputs = inputs.type(torch.cuda.FloatTensor)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                #preds = preds.reshape(preds.size(0),-1)
                loss = criterion(outputs, labels)
               # print(preds)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics 
            #running_loss += loss.item() * inputs.size(0)
            #print(labels.shape)
            a = preds.data
            
            #running_corrects = running_corrects + torch.sum(preds == labels.data)
            #total += labels.size(0)
            i = i+1
            '''if(i%100==0):
              print('loss :{} accuracy : {}'.format(running_loss/(i*32),running_corrects.double()/(total)))
              
              #print(i)
        epoch_loss = running_loss/(len(dataloaders[phase])*32)
        epoch_acc = running_corrects.double()/(len(dataloaders[phase])*32)
        '''
          #writer.add_scalar('./sanchit/scalar1', epoch_loss , epoch)
        print('{} Loss: {:.4f} , acc: {:.4f}'.format(phase, epoch_loss , epoch_acc))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    best_acc = epoch_acc
    print('Best val Acc: {:4f}'.format(best_acc))
    torch.save(resnet50, '/content/genreweights-{}.h5'.format(epoch_acc))
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(resnet50, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=5)