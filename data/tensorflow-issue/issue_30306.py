import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.layers import AvgPool3D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Concatenate, Add
from tensorflow.keras.estimator import model_to_estimator
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.utils import Sequence
import tensorflow as tf


def build_model(input_shape=(128, 128, 50, 1), n_class=3, multilabel=False):
   
    def spatial_reduction_block(inputs, block_name):
        filters = inputs._shape_as_list()[-1]
        with tf.name_scope(block_name):
            maxpool = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(inputs)
            conv_a_0 = Conv3D(filters=filters//4, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu')(inputs)
            conv_b_0 = Conv3D(filters=filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(inputs)
            conv_c_0 = Conv3D(filters=filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(inputs)

            conv_b_1 = Conv3D(filters=(5*filters)//16, kernel_size=(3, 3, 3), strides=(2, 2, 2), 
                              padding='same', activation='relu')(conv_b_0)
            conv_c_1 = Conv3D(filters=(5*filters)//16, kernel_size=(3, 3, 3), strides=(1, 1, 1), 
                              padding='same', activation='relu')(conv_c_0)
            conv_c_2 = Conv3D(filters=(7*filters)//16, kernel_size=(3, 3, 3), strides=(2, 2, 2), 
                              padding='same', activation='relu')(conv_c_1)

            concat_output = Concatenate()([maxpool, conv_a_0, conv_b_1, conv_c_2])

        return concat_output

    def residual_convolution_block(inputs, block_name):
        filters = inputs._shape_as_list()[-1]
        with tf.name_scope(block_name):
            conv_a_0 = Conv3D(filters=filters//2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(inputs)
            conv_b_0 = Conv3D(filters=filters//2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(inputs)
            conv_c_0 = Conv3D(filters=filters//2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(inputs)

            conv_b_1 = Conv3D(filters=filters//2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(conv_b_0)
            conv_c_1 = Conv3D(filters=filters//2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(conv_c_0)
            conv_c_2 = Conv3D(filters=filters//2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(conv_c_1)

            concat_output = Concatenate()([conv_a_0, conv_b_1, conv_c_2])

            conv_d_0 = Conv3D(filters=filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(concat_output)

            add_1 = Add()([conv_d_0, inputs])

        return add_1
    
    if not multilabel:
        activation_fn = 'softmax'
    else:
        activation_fn = 'sigmoid'
    
    inputs = Input(shape=input_shape, name='inputs')
    conv_1 = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(inputs)
    spatial_reduction_block_1 = spatial_reduction_block(conv_1, 'spatial_reduction_block_1')
    residual_convolution_block_1 = residual_convolution_block(spatial_reduction_block_1, 'residual_convolution_block_1')
    spatial_reduction_block_2 = spatial_reduction_block(residual_convolution_block_1, 'spatial_reduction_block_2')
    residual_convolution_block_2 = residual_convolution_block(spatial_reduction_block_2, 'residual_convolution_block_2')
    conv_2 = Conv3D(filters=512, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(residual_convolution_block_2)
    maxpool_1 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')(conv_2)
    conv_3 = Conv3D(filters=1024, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(maxpool_1)
    maxpool_2 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')(conv_3)
    flatten = Flatten()(maxpool_2)
    dropout_1 = Dropout(rate=0.2)(flatten)
    dense_1 = Dense(512, activation='sigmoid')(dropout_1)
    dropout_2 = Dropout(rate=0.2)(dense_1)
    outputs = Dense(n_class, activation=activation_fn, name='outputs')(dropout_2)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_model((128,128,50, 1), 3, False)

class mygenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, augment=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augment
    
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        x = [read_image(filename, self.augment) for filename in batch_x] # read a numpy array named filename
        y = [read_label(label) for label in batch_y]
        
        return np.array(x), np.array(y)

test_generator = mygenerator(X_TEST, Y_TEST, eval_batch_size, augment=False)

preds = model.predict_generator(test_generator, verbose=1, use_multiprocessing=True, steps=eval_steps)

use_multiprocessing