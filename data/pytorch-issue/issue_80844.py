#imports
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torchvision.transforms import transforms
import torch.nn.functional as F
device = torch.device('mps')


# COMMAND ---------

#download the dataset(s)
train_dataset = torchvision.datasets.MNIST('/Users/xy/work/content',train=True,
                                          transform=transforms.ToTensor(),download=True)

test_dataset = torchvision.datasets.MNIST('/Users/xy/work/content',train=False,
                                          transform=transforms.ToTensor(),download=True)

# COMMAND ----------

#create dataloaders from datasets
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                  batch_size=64,shuffle=True)

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                  batch_size=64,shuffle=True)

# COMMAND ----------

#plot first 10 images
images, labels = next(iter(train_dataloader))


#plt.figure(figsize=(10,8))
#for i in range(10):
  #  plt.subplot(2,5,i+1)
 #   plt.imshow(images[i][0],cmap='BuPu')
#plt.show()

# COMMAND ----------

#our first neural network
class MNIST_model(nn.Module):
    def __init__(self):
        super(MNIST_model,self).__init__()
        self.hidden_layer_1=nn.Linear(28*28,128)
        self.hidden_layer_2=nn.Linear(128,64)
        self.output_layer=nn.Linear(64,10)
    def forward(self,x):
        x = F.relu(self.hidden_layer_1(x)) 
        x = F.relu(self.hidden_layer_2(x))
        x = self.output_layer(x)
        return x

# COMMAND ----------

#instantiate the model and print paramters
model=MNIST_model()
model.to(device)

print(model.parameters)

# COMMAND ----------

#print dataset shape, reshape dataset to the desired format
print(images.shape)
images = images.view(images.shape[0], -1)
print(images.shape)

# COMMAND ----------

#pass images (data) trough our model:
logps = model(images.to(device)) #log probabilities
print(logps.shape) # print output shape
print(logps) # print output

# COMMAND ----------

#create a loss function and calculate a loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logps.detach().cpu(), labels) #calculate the NLL loss

# COMMAND ----------

#training loop
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in train_dataloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        logits = model(images.to(device))
        #print(logits)
        loss = criterion(logits, labels.to(device))
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e + 1, running_loss/len(train_dataloader)))


# COMMAND ----------

#validate our model
correct_count, all_count = 0, 0
for images,labels in test_dataloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    print(img)
    with torch.no_grad():
        logps = model(img.to(device))

    #print(logps)
    ps = torch.exp(logps)
    probab = list(ps.detach().cpu().numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    #print(str(pred_label) + " " + str(true_label))
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

# COMMAND ----------

import numpy as np
def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

images, labels = next(iter(test_dataloader))

img = images[0].view(1, 784)
with torch.no_grad():
    logps = model(img.to(device))

    
ps = torch.exp(logps)
probab = list(ps.detach().cpu().numpy()[0])
data = probab
norm = (data - np.min(data)) / (np.max(data) - np.min(data))

print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), norm)

# COMMAND ----------