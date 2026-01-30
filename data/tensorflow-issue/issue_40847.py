from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

#create the model
model = Model(input_img,d1)

#review the model   
model.summary()

#Compile the model
model.compile(optimizer=optimizer,loss=theLoss,metrics =['accuracy'])

#save the only the best weights acheived during training
filepath='weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callbacks_list=[checkpoint]

y_train = to_categorical(y_train,num_classes=2)
X_train = X_train

#fit model
model.fit(X_train,y_train,validation_split=(0.15),epochs=1,batch_size=2,verbose=1,callbacks=callbacks_list,shuffle=True)

import tensorflow as tf
print(tf.__version__)
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_normal, VarianceScaling,he_normal
from tensorflow.keras.utils import to_categorical, HDF5Matrix
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalMaxPooling2D, Dropout, Conv2D,Activation,MaxPooling2D,Add
from tensorflow.keras.layers import BatchNormalization, concatenate, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras import regularizers

tf.compat.v1.reset_default_graph()
K.clear_session()

#hyperparams
input_img = Input(shape=(512,512,1))
ch1=32
ch2=ch1*2
ch3=ch2*2
ch4 = ch3*2
ch5 = ch4*2
ks=3
rg = 0.01
init = glorot_normal(seed=0)
activation = 'relu'
optimizer = Adam(learning_rate=0.00001)
theLoss = 'categorical_crossentropy'

#model layers
xin = Conv2D(ch1,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(input_img)
x1 = Conv2D(ch1,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(xin)
x1 = Activation(activation)(x1)
x1 = BatchNormalization(axis=-1)(x1)
x1 = Add()([xin,x1])
x1 = MaxPooling2D((2,2),padding='same')(x1) #image size = 256

xin = Conv2D(ch2,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x1)
x2 = Conv2D(ch2,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(xin)
x2 = Activation(activation)(x2)
x2 = BatchNormalization(axis=-1)(x2)
x2 = Add()([xin,x2])
x2 = MaxPooling2D((2,2),padding='same')(x2) #image size = 128

xin = Conv2D(ch3,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x2)
x3 = Conv2D(ch3,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(xin)
x3 = Activation(activation)(x3)
x3 = BatchNormalization(axis=-1)(x3)
x3 = Add()([xin,x3])
x3 = MaxPooling2D((2,2),padding='same')(x3) #image size = 64

xin = Conv2D(ch4,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x3)
x4 = Conv2D(ch4,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(xin)
x4 = Activation(activation)(x4)
x4 = BatchNormalization(axis=-1)(x4)
x4 = Add()([xin,x4])
x4 = MaxPooling2D((2,2),padding='same')(x4) #image size = 32

xin = Conv2D(ch5,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x4)
x5 = Conv2D(ch5,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(xin)
x5 = Activation(activation)(x5)
x5 = BatchNormalization(axis=-1)(x5)
x5 = Add()([xin,x5])

encoded = MaxPooling2D((2,2),padding='same')(x5) #image size = 16
xu = Conv2D(ch5,(1,1),padding='valid')(encoded)

x6 = concatenate([xu,encoded],axis=-1)
x6 = UpSampling2D((2,2), interpolation='bilinear')(x6)                  #image size = 32
x6 = Conv2D(ch5,kernel_size=(1,1),padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x6)
x6 = Conv2D(ch5,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x6)
x6 = Conv2D(ch5,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x6)
x6 = Activation(activation)(x6)
x6 = BatchNormalization(axis=-1)(x6)


x7 = concatenate([x6,x4],axis=-1)
x7 = UpSampling2D((2,2), interpolation='bilinear')(x7)                  #image size =  64
x7 = Conv2D(ch4,kernel_size=(1,1),padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x7)
x7 = Conv2D(ch4,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x7)
x7 = Conv2D(ch4,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x7)
x7 = Activation(activation)(x7)
x7 = BatchNormalization(axis=-1)(x7)


x8 = concatenate([x7,x3],axis=-1)
x8 = UpSampling2D((2,2), interpolation='bilinear')(x8)                  #image size = 128
x8 = Conv2D(ch3,kernel_size=(1,1),padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x8)
x8 = Conv2D(ch3,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x8)
x8 = Conv2D(ch3,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x8)
x8 = Activation(activation)(x8)
x8 = BatchNormalization(axis=-1)(x8)


x9 = concatenate([x8,x2],axis=-1)
x9 = UpSampling2D((2,2), interpolation='bilinear')(x9)                  #image size = 256
x9 = Conv2D(ch2,kernel_size=(1,1),padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x9)
x9 = Conv2D(ch2,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x9)
x9 = Activation(activation)(x9)
x9 = BatchNormalization(axis=-1)(x9)


x10 = concatenate([x9,x1],axis=-1)
x10 = UpSampling2D((2,2), interpolation='bilinear')(x10)                  #image size = 512
x10 = Conv2D(ch1,kernel_size=(1,1),padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x10)
x10 = Conv2D(ch1,ks,padding='same',use_bias=1,kernel_initializer=init,kernel_regularizer=regularizers.l2(rg),bias_initializer='zeros')(x10)
x10 = Activation(activation)(x10)
x10 = BatchNormalization(axis=-1)(x10)

decoded = Conv2D(2,(1,1),padding='valid',use_bias=1)(x10)
d1 = Activation('softmax')(decoded)

#create the model
model = Model(input_img,d1)

#review the model   
model.summary()

#Compile the model
model.compile(optimizer=optimizer,loss=theLoss,metrics =['accuracy'])

#save the only the best weights acheived during training
filepath='weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callbacks_list=[checkpoint]

#fit the model
y_train = to_categorical(y_train,num_classes=2)
X_train=X_train
model.fit(X_train,y_train,validation_split=(0.15),epochs=1,batch_size=2,verbose=1,callbacks=callbacks_list,shuffle=True)