from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import keras
from keras.layers import Activation, Input, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
import scipy.io as sio
import numpy as np
from keras.models import load_model,Model

first_model = load_model('first_model .hdf5')
first_model.name='first_model'

for ii in range(11):
    exec("input_1_"+str(ii)+"=Input(shape=(1771,))")
    exec('output_1_'+str(ii)+'=first_model(input_1_'+str(ii)+')')

concatenated = keras.layers.concatenate([output_1_0, output_1_1, output_1_2, output_1_3, output_1_4, output_1_5,
                                         output_1_6, output_1_7, output_1_8, output_1_9, output_1_10],name='concat')

second_model = load_model('speech(noisy_to_s).hdf5')
second_model.name='second_model'

x=second_model(concatenated) 

model = Model(inputs=[input_1_0, input_1_1, input_1_2, input_1_3, input_1_4, input_1_5, 
                       input_1_6, input_1_7, input_1_8, input_1_9, input_1_10],outputs=[x])

batch_size = 1024
Adam = keras.optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=Adam, metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='save.hdf5',
                               monitor='val_loss', verbose=1, save_best_only=True)
history=model.fit([x_train[0:-10,:],x_train[1:-9,:],x_train[2:-8,:],x_train[3:-7,:],x_train[4:-6,:],x_train[5:-5,:],
                   x_train[6:-4,:],x_train[7:-3,:],x_train[8:-2,:],x_train[9:-1,:],x_train[10:,:]],y_train[10:,:],
                  batch_size=batch_size, epochs=200,verbose=0,
                  validation_data=([x_valid[0:-10,:],x_valid[1:-9,:],x_valid[2:-8,:],x_valid[3:-7,:],x_valid[4:-6,:],
                                    x_valid[5:-5,:],x_valid[6:-4,:],x_valid[7:-3,:],x_valid[8:-2,:],x_valid[9:-1,:],
                                    x_valid[10:,:]],y_valid[10:,:]),
                  callbacks=[checkpointer])

model = load_model(checkpointer.filepath)