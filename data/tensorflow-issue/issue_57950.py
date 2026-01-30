import random
from tensorflow.keras import optimizers

import kerastuner as kt
#from google.colab import drive
import pandas as pd
import glob
import pdb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import subprocess
import h5py
from tensorflow.keras import Sequential, layers, Model


#drive.mount("/content/gdrive")
data_path=r'C:\\Users\\q75714hz\\New folder\\UVLIF\\PLAIR_HK\\Processed\\'

"""1) Define custom generators that will load the data from multiple CSV files in batches during the training phase. """

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    CONTAINS SPECIFIC INFO FOR AUTOENCODERS
    """
    def __init__(self, list_files, to_fit=True, mini_batch = 1000, batch_size=1, shuffle=True):
        """Initialization
        :param data_path: path to datafiles
        :param list_files: list of image labels (file names)
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param shuffle: True to shuffle label indexes after every epoch
        """

        # We have to create a mapping to the file name and the subset of data
        # extracted from that file as a dictionary or list.
        # To do this we need to count the number of lines in each file
        # and then divide that by the mini_batch and loop through each
        # chunck and define a starting point to extract the data

        self.list_files = list_files
        self.mini_batch = mini_batch
        self.data_path = data_path
        #self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        #self.dim = dim
        #self.n_channels = n_channels
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_files) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_files_temp = [self.list_files[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_files_temp)

        #return X

        if self.to_fit:
            y = X
            return (X, y)
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_files_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))


        if len(list_files_temp) == 1:
          path = list_files_temp[0][0]
          start_loc = list_files_temp[0][1]
          end_loc = list_files_temp[0][2]
          stop_point = min(start_loc+self.mini_batch,end_loc)

          #info_df=pd.read_csv(path,skiprows=start_loc+1,nrows=min(self.mini_batch,end_loc-start_loc))
          #Scattering_df = info_df.iloc[:, 34::][(info_df.iloc[:, 34::].T != 0).any()]
          #Scattering_df[Scattering_df < 0] = 0
          #Scattering_df=Scattering_df.div(Scattering_df.max(axis=1), axis=0)
          ## extract the numpy array and then reshape back to the original size.
          #images = Scattering_df.loc[:, Scattering_df.columns != 'label'].to_numpy()
          #X = np.reshape(images, (images.shape[0], 80, 24, 1))

          hf = h5py.File(path, 'r')
          data=hf['test']['block0_values'][start_loc:stop_point, 33::]

          #info_df=pd.read_hdf(filename, "test",start=start_loc,stop=stop_point)
          #data=info_df.iloc[:, 33::].to_numpy()
          data = data[~np.all(data == 0, axis=1)]
          data=data[~np.isnan(data).all(axis=1)]
          data=data[np.isfinite(data).all(axis=1)]

          # basic stratgey here is to convert the image into a sharpened replica
          data=data/np.max(data,axis=1)[:,None]
          #std=np.std(data,axis=1)[:,None]
          #mean=np.mean(data,axis=1)[:,None]
          #data[data >= 0.001]=1.0
          #data[data < 0.001]=0.0

          #X = info_df.iloc[:, 33::].to_numpy().reshape(info_df.shape[0],80,24,1)
          #data[data > 0.0001]=1.0
          X=data.reshape(data.shape[0],80,24,1)

        else:
          Scattering_list=[None] * len(list_files_temp)
          step=0
          for entry in list_files_temp:
            path = entry[0]
            start_loc = entry[1]
            end_loc = entry[2]
            stop_point = min(start_loc+self.mini_batch,end_loc)
            #info_df=pd.read_csv(path,na_filter=False,header=None,skiprows=start_loc+1,nrows=min(self.mini_batch,end_loc-start_loc))
            #Scattering_df = info_df.iloc[:, 34::][(info_df.iloc[:, 34::].T != 0).any()]
            #Scattering_df[Scattering_df < 0] = 0
            #Scattering_df=Scattering_df.div(Scattering_df.max(axis=1), axis=0)
            #Scattering_df = Scattering_df.dropna()
            #info_df=pd.read_hdf(filename, "test",start=start_loc,stop=stop_point)
            hf = h5py.File(path, 'r')
            Scattering_list[step]=hf['test']['block0_values'][start_loc:stop_point, 33::]
            #Scattering_list.append(info_df)
            step+=1
          #Scattering_df2=pd.concat(Scattering_list,axis=0) #pd.DataFrame.from_dict(Scattering_dict, orient='index')
          #extract the numpy array and then reshape back to the original size.
          #images = Scattering_df2.loc[:, Scattering_df2.columns != 'label'].to_numpy()
          #pdb.set_trace()
          #data=Scattering_df2.iloc[:, 33::].to_numpy()
          data = Scattering_list[~np.all(Scattering_list == 0, axis=1)]
          data=data[~np.isnan(data).all(axis=1)]
          data=data[np.isfinite(data).all(axis=1)]
          data=data/np.max(data,axis=1)[:,None]
          #data[data >= 0.001]=1.0
          #data[data < 0.001]=0.0
          #data[data > 0.0001]=1.0
          #X = Scattering_df2.iloc[:, 33::].to_numpy().reshape(Scattering_df2.shape[0],80,24,1)
          X=data.reshape(data.shape[0],80,24,1)

        return X

list_files = glob.glob(data_path+'*.hdf')

# define a minibatch which would normally be used in the standard training method
minibatch = 1000

list_of_mappings = []

for filename in list_files:
    # lines = int(subprocess.getoutput("sed -n '$=' " + filename))
    hf=pd.read_hdf(filename,mode='r')
    lines=int(hf.shape[0])
    #pdb.set_trace()
    chunks = int(np.ceil(lines / minibatch))
    for step in range(chunks):
        sublist=[]
        sublist.append(filename)
        sublist.append(step*minibatch)
        sublist.append(min((step + 1)*minibatch,lines-2))
        list_of_mappings.append(sublist)
print(list_of_mappings[0:10])
print(list_of_mappings[11:20])
# pdb.set_trace()

training_generator = DataGenerator(list_of_mappings[:])
validation_generator = DataGenerator(list_of_mappings[:])
print(len(list_of_mappings[:]))
print(len(training_generator))


# define tunner of ae

def model(hp):

    original_inputs = keras.Input(shape=(80, 24, 1), name='encoder_input')
    variance_scale = 0.3
    init = tf.keras.initializers.VarianceScaling(scale=variance_scale, mode='fan_in', distribution='uniform')
    layer= layers.Conv2D(filters=hp.Choice("num_filters_layer_1", values=[8, 32], default=8), kernel_size=3,
                               activation='relu', kernel_initializer=init, padding='same',
                               strides=1)(original_inputs)
    layer1 = layers.Conv2D(filters=hp.Int("num_filters_layer_2", min_value=16, max_value=64, step=16), kernel_size=3,
                          activation='relu', kernel_initializer=init, padding='same',
                          strides=1)(layer)
    layer2 = layers.Conv2D(filters=hp.Int("num_filters_layer_3", min_value=16, max_value=96, step=16), kernel_size=3,
                          activation='relu', kernel_initializer=init, padding='same',
                          strides=1)(layer1)
    layer3 = layers.Conv2D(filters=hp.Int("num_filters_layer_4", min_value=16, max_value=112, step=16), kernel_size=3,
                          activation='relu', kernel_initializer=init, padding='same',
                          strides=1)(layer2)



    layer_flatten = layers.Flatten()(layer3)
    h = layers.Dense(hp.Int("num_Dense", 0, 600, 200), activation='relu', name="encoding_5")(layer_flatten)
    latent_layer = layers.Dense(hp.Int("latent_space", 20, 40, 10), activation='relu')(h)

    #decoder
    latent_inputs_cnn = keras.Input(shape=(latent_layer.shape[1],), name='latent_input')
    dec_layer1_cnn = layers.Dense(h.shape[1], activation='relu')(latent_inputs_cnn)
    dec_layer2_cnn = layers.Dense(layer_flatten.shape[1], activation='relu')(dec_layer1_cnn)
    dec_layer = layers.Reshape((layer3.shape[1], layer3.shape[2], layer3.shape[3]))(dec_layer2_cnn)

    dec_layer3_cnn = layers.Conv2DTranspose(hp.Int("num_filters_layer_3", min_value=16, max_value=96, step=16),
                                            kernel_size=3, activation='relu', kernel_initializer=init,
                                            padding='same', strides=1)(dec_layer)

    dec_layer4_cnn = layers.Conv2DTranspose(filters=hp.Int("num_filters_layer_2", min_value=16, max_value=64, step=16), kernel_size=3,
                           activation='relu', kernel_initializer=init, padding='same',
                           strides=1)(dec_layer3_cnn)


    dec_layer5_cnn=layers.Conv2DTranspose(filters=hp.Choice("num_filters_layer_1",values=[8,32],default=8), kernel_size=3, activation='relu', kernel_initializer=init,
                                            padding='same', strides=1)(dec_layer4_cnn)
    dec_layer6_cnn = layers.Conv2DTranspose(original_inputs.shape[3], (3, 3), activation='sigmoid',
                                            kernel_initializer=init, padding='same', strides=1)(dec_layer5_cnn)
    dec_cnn = Model(inputs=latent_inputs_cnn, outputs=dec_layer6_cnn, name='decoder_cnn')
    outputs = dec_cnn(latent_layer)

    cnn_ae = Model(inputs=original_inputs, outputs=outputs, name='cnn_ae')

    cnn_ae.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
                   loss='binary_crossentropy', metrics=['accuracy'])

    return cnn_ae

tuner = kt.Hyperband(model,
                     objective='loss',
                     max_epochs=5,
                     factor=3,
                     directory='my_dir_SE',
                     project_name='intro_to_kt_se' ,overwrite=True)



tuner.search(training_generator, epochs=4, workers=4)

# Get the optimal hyperparameters
best_hps =tuner.get_best_hyperparameters(num_trials=1)[0]


# Now we can train the AE and save it, if needs be, for later use.
# Build the model with the optimal hyperparameters and train it on the data for 30 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(training_generator, epochs=40, workers=4)
model.save('ae_sequence_scattering_40epochs.h5')
val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
hypermodel = tuner.hypermodel.build(best_hps)
# Retrain the model
history_new = hypermodel.fit(training_generator, epochs=best_epoch,workers=4)
hypermodel.save('ae_sequence_scattering.h5')