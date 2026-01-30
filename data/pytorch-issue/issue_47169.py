import numpy as np
#torch.autograd.set_detect_anomaly(True)
#REBUILD_DATA = True # set to true to one once, then back to false unless you want to change something in your training data.
import time
start_time = time.time()
import os
import cv2
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import ConfusionMatrixDisplay# Build the confusion matrix of our 2-class classification problem  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score # for evaluating the model
import torch # PyTorch libraries and modules
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from sklearn.metrics import classification_report
import seaborn as sns # color of graphe
'''from  owlready2 import *
import textdistance
'''
import numpy as np



#Channels ordering : first channel (taille , shape of each element ) to ==> last channel ( shape, size )
def changechannel(data, a, b):
    data = np.asarray(data)
    data = np.rollaxis(data, a, b)
    return(data)

# convert (240,240,155) to ======> (120, 120, 120) #
def resize3Dimages(data):
    train_x = []
    for i in range(len(data)):
        image = data[i] 
        width = 120
        height = 120
        img_zeros = np.zeros((len(image), width, height))

        for idx in range(len(image)):
            img = data[i][idx, :, :]
            img_sm = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC) 
            img_zeros[idx, :, :] = img_sm
#  convert (240,120,120) to ======> (120,120,120)
        img_zeros = img_zeros[::2, :, :] ### 240/2 =120 
        train_x.append(img_zeros)
############################# save images in list ################################
    return(np.asarray(train_x)) ## convert list to nd array 
# end ...
# 1 channel to 3 channel 
def channel1to3 (data): 
    data = np.stack((data,) * 3, axis=-1)
    return(data)
print(" preprocessing  --- %s seconds ---" % (time.time() - start_time))

print('Building of CNN')
import os
# for reading and displaying images
import matplotlib.pyplot as plt
# for evaluating the model
from sklearn.metrics import accuracy_score
# PyTorch libraries and modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
###################### to use after 
num_classes = 2

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__() 
        
        self.conv_layer1 = self._conv_layer_set(3, 32) 
        self.conv_layer2 = self._conv_layer_set(32, 64) 
        self.conv_layer3 = self._conv_layer_set(64, 128)
        self.conv_layer4 = self._conv_layer_set(128, 256)
        self.conv_layer5 = self._conv_layer_set(256, 512)

        self.fc1 = nn.Linear(512, 128) 
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.6, inplace = True)   
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.ReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        out = out.view(out.size(0), -1)
        #print('conv shape', out.shape)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out) # batchnormalization pour 
        out = self.drop(out)
        out = self.fc2(out)
        #out = F.softmax(out, dim=1)
        return out
#Definition of hyperparameters
n_iters = 2
num_epochs =100
# Create CNN
model = CNNModel()
model.cuda() #pour utiliser   GPU
print(model)
# Cross Entropy Loss 
for param in model.parameters():
    param.requires_grad = True 
    error = nn.CrossEntropyLoss()
# SGD Optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

###################################################accuracy function ##################################
def accuracyCalc (predicted, targets):
    correct = 0
    p = predicted.tolist()
    t = targets.flatten().tolist() 
    for i in range(len(p)):
        if (p[i] == t[i]):
            correct +=1
    accuracy = 100 * correct / targets.shape[0]
    #print("correct: ",correct)
    return(accuracy)
#######################################################################################################
print(" build model --- %s seconds ---" % (time.time() - start_time))
#######################################################{{{{{{{training}}}}}}}##################################
print('data preparation ')
training_data = np.load("Datasets/brats/Train/training_data.npy", allow_pickle=True)
targets = np.load("Datasets/brats/Train/targets.npy", allow_pickle=True)
from sklearn.utils import shuffle
training_data, targets = shuffle(training_data, targets)

training_data = changechannel(training_data, 1, 5) #Channels ordering : first channel to ==> last channel'
training_data  = resize3Dimages(training_data) #resize images
training_data = channel1to3(training_data,)#1 channel to 3 channel ===> RGB
training_data = changechannel(training_data, 4, 1)# last to first

loss_list_train = []
accuracy_list_train = []
for epoch in range(num_epochs): 
    outputs = []
    outputs = torch.tensor(outputs, requires_grad=True)  
    outputs= outputs.clone().detach().cuda()
    for fold in range(0, len(training_data), 5): 
        xtrain = training_data[fold : fold+5]
        xtrain = torch.tensor(xtrain, requires_grad=True).clone().detach().float().cuda() # deplacer l'inpt vers GPU 
        xtrain = xtrain.view(5, 3, 120, 120, 120) 
        # Clear gradients
        # Forward propagation
        optimizer.zero_grad() 
        v = model(xtrain)
        v = v.clone().detach().requires_grad_(True) 
        outputs = torch.cat((outputs,v),dim=0)
    targets = torch.Tensor(targets).clone().detach()
    labels = targets.cuda()
    outputs =outputs.clone().detach().requires_grad_(True) 
    _, predicted = torch.max(outputs, 1) 
    accuracy = accuracyCalc(predicted, targets)
    labels = labels.long() 
    labels=labels.view(-1) 
    loss = nn.CrossEntropyLoss()
    loss = loss(outputs, labels)    
    # Calculating gradients
    loss.backward()
    # Update parameters
    optimizer.step()
    loss_list_train.append(loss.clone()) #loss values loss
    accuracy_list_train.append(accuracy/100)
    np.save('Datasets/brats/accuracy_list_train.npy', np.array(accuracy_list_train))
    np.save('Datasets/brats/loss_list_train.npy', np.array(loss_list_train))
    print('Iteration: {}/{}  Loss: {}  Accuracy: {} %'.format(epoch+1,  num_epochs, loss.clone(), accuracy))
print('Model training  : Finished')