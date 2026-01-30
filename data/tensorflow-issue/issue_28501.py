import random
from tensorflow.keras import layers

from tensorflow import keras
import numpy as np

L = keras.layers

embedding_matrix = np.random.random((10, 5))

model = keras.Sequential([
    L.Embedding(input_dim=10, 
                output_dim=5,
                weights=[embedding_matrix],
                trainable=False)
])

model.compile('rmsprop', 'mse')

embedding_matrix[2] == model.predict([2])[0][0]

array([False, False, False, False, False])

# 1.13.1
array([ True,  True,  True,  True,  True])
# 2.0 alpha
array([False, False, False, False, False])