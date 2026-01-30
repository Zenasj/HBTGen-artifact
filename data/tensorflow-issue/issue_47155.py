import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# Train data preparation
N = datasets[0].shape[0]
conv_input_width = W.shape[1]
conv_input_height = int(datasets[0].shape[1]-1)

# For each word write a word index (not vector) to X tensor
train_X = np.zeros((N, conv_input_height), dtype=np.int)
train_Y = np.zeros((N, 2), dtype=np.int)
for i in range(N):
    for j in range(conv_input_height):
        train_X[i, j] = datasets[0][i, j]
    
print ('train_X.shape = {}'.format(train_X.shape))
print ('train_Y.shape = {}'.format(train_Y.shape))

# Validation data preparation
Nv = datasets[1].shape[0]

# For each word write a word index (not vector) to X tensor
val_X = np.zeros((Nv, conv_input_height), dtype=np.int)
val_Y = np.zeros((Nv, 2), dtype=np.int)
for i in range(Nv):
    for j in range(conv_input_height):
        val_X[i, j] = datasets[1][i, j]
print('val_X.shape = {}'.format(val_X.shape))
print('val_Y.shape = {}'.format(val_Y.shape))
for i in range(Nv):
    val_Y[i,data_train.iloc[i,3]] = 1

from keras.optimizers import RMSprop
from keras import backend
backend.set_image_data_format('channels_first')
import keras


# Number of feature maps (outputs of convolutional layer)
N_fm = 200
# kernel size of convolutional layer
kernel_size = 5

model = Sequential()
# Embedding layer (lookup table of trainable word vectors)
model.add(Embedding(input_dim=W.shape[0], 
                    output_dim=W.shape[1], 
                    input_length=conv_input_height,
                    weights=[W], 
                    embeddings_constraint=UnitNorm,
                    name = 'e_l'))
# Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
model.add(Reshape((1, conv_input_height, conv_input_width)))

# first convolutional layer
model.add(Convolution2D(N_fm,
                        kernel_size, 
                        conv_input_width,
                        kernel_initializer='random_uniform',
                        padding='valid',
                        kernel_regularizer=l2(0.001)))
# ReLU activation
model.add(Activation('relu'))

# aggregate data in every feature map to scalar using MAX operation
model.add(MaxPooling2D(pool_size=(conv_input_height+kernel_size+1,1), padding='same'))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(128,kernel_initializer='random_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.4))
# Inner Product layer (as in regular neural network, but without non-linear activation function)
model.add(Dense(2))
# SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
model.add(Activation('softmax'))

# Custom optimizers could be used, though right now standard adadelta is employed
opt = RMSprop(lr=0.001, rho=0.9, epsilon=None)
model.compile(loss='mean_squared_error', 
              optimizer=opt,
              metrics=['accuracy'])

N_epoch = 3

for i in range(N_epoch):
    model.fit(x=train_X,y=train_Y,batch_size=32,epochs=10,verbose=1, validation_data=(val_X,val_Y))
    output = model.predict_proba(val_X, batch_size=10, verbose=1)
    # find validation accuracy using the best threshold value t
    vacc = np.max([np.sum((output[:,1]>t)==(val_Y[:,1]>0.5))*1.0/len(output) for t in np.arange(0.0, 1.0, 0.01)])
    # find validation AUC
    vauc = roc_auc_score(val_Y, output)
    val_acc.append(vacc)
    val_auc.append(vauc)
    print('Epoch {}: validation accuracy = {:.3%}, validation AUC = {:.3%}'.format(epoch, vacc, vauc))
    epoch += 1
    
print('{} epochs passed'.format(epoch))
print('Accuracy on validation dataset:')
print(val_acc)
print('AUC on validation dataset:')
print(val_auc)

X = np.random.randint(100, size=(32, 10))
Y = np.ones((32, 1))
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(100, 8, input_length=10, embeddings_constraint=tf.keras.constraints.UnitNorm(axis=1)))
model.add(tf.keras.layers.Dense(1))
model.compile('rmsprop', 'mse')
model.fit(X, Y)

X = np.random.randint(100, size=(32, 10))
Y = np.ones((32, 1))
input = tf.keras.Input(shape=(10,))
output = tf.keras.layers.Embedding(100, 8, input_length=10)(input)
output = tf.keras.constraints.UnitNorm(axis=1)(output)
output = tf.keras.layers.Dense(1)(output)
model = tf.keras.Model(input, output)
model.compile('rmsprop', 'mse')
model.fit(X, Y)