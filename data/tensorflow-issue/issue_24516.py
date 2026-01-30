from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy.random as rng
from whalegenerator import WhaleGenerator

import tensorflow as tf
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential, save_model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Conv2D, Lambda, Subtract, Dense, Flatten,MaxPooling2D

ProcessImages = True
TrainPath = '.\\processed\\train'
ProcessedPath = '.\\processed'
TrainTruthPath = '.\\train.csv'

def buildModel():
    #We are building a saimese network, whaling problem should use comparison
    input_shape = (256,256,1)
    left = Input(input_shape)
    right = Input(input_shape)

    convnet = Sequential()
    convnet.add(Conv2D(64,(9,9),activation='relu',input_shape=input_shape,kernel_initializer='random_normal',kernel_regularizer=l2(2E-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(7,7),activation='relu',kernel_regularizer=l2(2E-4),kernel_initializer='random_normal',bias_initializer='random_normal'))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256,(5,5),activation='relu',kernel_initializer='random_normal',kernel_regularizer=l2(2e-4),bias_initializer='random_normal'))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='random_normal',kernel_regularizer=l2(2e-4),bias_initializer='random_normal'))
    convnet.add(Flatten())
    convnet.add(Dense(2048,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer='random_normal',bias_initializer='random_normal'))

    encodedL = convnet(left)
    encodedR = convnet(right)

    subtract = Subtract()([encodedL,encodedR])
    diff = Lambda(lambda x: K.abs(x))(subtract)
    prediction = Dense(1,activation='sigmoid',bias_initializer='random_normal')(diff)
    siamese_net = Model(inputs=[left,right],outputs=prediction)
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    siamese_net.compile(loss="binary_crossentropy",optimizer=Adam(6E-5),options=run_opts,metrics=['accuracy'])
    return siamese_net 

model = buildModel()
print(model.count_params())
batch_size = 32
generator = WhaleGenerator(TrainPath,TrainTruthPath,batch_size,(256,256))

try:
        result = model.fit_generator(generator=generator,epochs=100,steps_per_epoch=int(np.floor(len(generator)/batch_size)),max_queue_size=50,verbose=2,callbacks=[
                ModelCheckpoint('.\\models\\whale_256.h5',save_best_only=True,monitor='accuracy')
                ])
except ValueError as e:
        print(e)
except Exception as e:
        print(e)
else:
        print('unkown error')

print('finished!')

import os
import cv2
import random
import numpy as np
import pandas as pd
from keras.utils import Sequence

class WhaleGenerator(Sequence):

    #Number of items in the group
    def __len__(self):
        return int(np.floor(len(self.imageList)/2))

    def __getitem__(self,index):
        X = [np.empty((self.batch_size, *self.dim, 1)) for i in range(2)]
        Y = np.empty((self.batch_size), dtype=float)

        # Generate data
        matches = 0
        matchKeys = list(self.hasMatches.items())
        for i in range(self.batch_size):
            if self.batch_size - i < self.min_match and matches < self.min_match:
                id1 = random.choice(matchKeys)[0]
                images1 = self.imgGroups[id1]
                X[0][i,] = images1[random.randint(0,len(images1)-1)]
                X[1][i,] = images1[random.randint(0,len(images1)-1)]
                Y[i] = 1.0
                matches += 1
            else:
                item1 = self.imageList[random.randint(0,len(self.imageList)-1)]
                id1 = item1["Id"]
                X[0][i,] = item1["Image"]
                images1 = self.imgGroups[id1]
                if id1 != "new_whale" and len(images1) > 1 and random.random() <= 0.5: 
                    X[1][i,] = images1[random.randint(0,len(images1)-1)]
                    Y[i] = 1.0
                    matches += 1.0
                else:
                    item2 = self.imageList[random.randint(0,len(self.imageList)-1)]
                    id2 = item2["Id"]
                    X[1][i,] = item2["Image"]
                    if id1 != "new_whale" and id1 == id2:
                        Y[i] = 1.0
                        matches += 1.0
                    else:
                        Y[i] = 0.0

        return X, Y

    def __init__(self, image_path, csv, batch_size, dim):

        self.min_match = 3
        self.imgGroups = { }
        self.imageList = [ ]
        self.hasMatches = { }
        self.dim = dim
        self.df = pd.read_csv(csv)
        self.batch_size = batch_size

        for i, row in self.df.iterrows():

            _id = row["Id"]
            img = cv2.imread(os.path.join(image_path,row["Image"]))

            if img is None:
                continue

            img = img[:,:,0]
            img = img.reshape((*self.dim,1))
            self.imageList.append({ "Id": _id, "Image": img })
            if not _id in self.imgGroups:
                self.imgGroups[_id] = [img]
            else:
                if _id != "new_whale":
                    self.hasMatches[_id] = True

                self.imgGroups[_id].append(img)