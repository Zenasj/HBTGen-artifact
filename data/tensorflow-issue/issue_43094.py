from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, losses
from tensorflow.keras.layers import Input, Dense, Flatten



def make_model(input_shape, output_shape, batch_size=32):
    model = Sequential()
    model.add(Input(shape=input_shape, batch_size=batch_size, name='input'))
    
    model.add(Flatten())
    model.add(Dense(output_shape))
    model.build((batch_size, *input_shape))
    return model
    

if __name__ == '__main__':
    batch_size = 32
    input_size = (5, 5)
    output_size = 1
    num_samples = 100
    model = make_model(input_size, output_size, batch_size=batch_size)
    model.compile(optimizer=optimizers.Adam(), loss=losses.SparseCategoricalCrossentropy(from_logits=True))       
    X = np.ones((num_samples, *input_size))
    y = np.zeros(num_samples)
    model.fit(X, y, batch_size=batch_size, epochs=10)