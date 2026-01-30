import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import CuDNNLSTM, Dense
from tensorflow.keras.optimizers import Adadelta


feature_count = 25
batch_size = 16
look_back = 5
target_groups = 10

def random_data_generator( ):
    x_data_size =(batch_size, look_back, feature_count) # batches, lookback, features
    x_data = np.random.uniform(low=-1.0, high=5, size=x_data_size)
 
    y_data_size = (batch_size, target_groups)
    Y_data = np.random.randint(low=1, high=21, size=y_data_size)
    
    return x_data, Y_data
 
def get_simple_Dataset_generator():        
    while True:
        yield random_data_generator()

def build_model():
    model = Sequential()
    model.add(CuDNNLSTM(feature_count,
                    batch_input_shape=(batch_size,look_back, feature_count),
                    stateful=False))  
    model.add(Dense(target_groups, activation='softmax'))
    optimizer = Adadelta(learning_rate=1.0, epsilon=None) 
    model.compile(loss='categorical_crossentropy', optimizer=optimizer) 
    return model


def run_training():
   
    model = build_model()
    train_generator = get_simple_Dataset_generator()
    validation_generator = get_simple_Dataset_generator()
    class_weights = {0:2, 1:8, 2:1, 3:4, 4:8, 5:35, 6:30, 7:4, 8:5, 9:3}

    model.fit_generator(generator = train_generator,
            steps_per_epoch=1,
            epochs=1000,            
            verbose=2,
            validation_data=validation_generator,
            validation_steps=20,
            max_queue_size = 10,
            workers = 0, 
            use_multiprocessing = False,
            class_weight = class_weights
            )

if __name__ == '__main__': 
    run_training()