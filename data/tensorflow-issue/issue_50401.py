import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#input in general has shape (N_inputs, variable length, N_input_channels)    
X = [[[4.,3,2],[2,1,3],[-1,2,1]],
     [[1,2,3],[3,2,4]]]
X = tf.ragged.constant(X, ragged_rank=1, dtype=tf.float64)

#output in general has shape (N_inputs, variable but same as corresponding input, N_classification_classes)
Y = [[[0,0,1],[0,1,0],[1,0,0]],
     [[0,0,1],[1,0,0]]]
Y = tf.ragged.constant(Y, ragged_rank=1)

#Documentation says for temporal data we can pass 2D array with shape (samples, sequence_length) 
weights = [[100,1,1],
           [100,1]]
weights = np.array(weights)

model = SimpleModel(width=16, in_features=3, out_features=3)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X,Y) #works fine
model.fit(X,Y, sample_weight=weights) #throws error

#input in general has shape (N_inputs, 2, N_input_channels)    
X = [[[4.,3,2],[2,1,3]],
     [[1,2,3],[3,2,4]]]
X = tf.constant(X, dtype=tf.float64)

#output in general has shape (N_inputs, 2, N_classification_classes)
Y = [[[0,0,1],[0,1,0]],
     [[0,0,1],[1,0,0]]]
Y = tf.constant(Y)

#Documentation says for temporal data we can pass 2D array with shape (samples, sequence_length) 
weights = [[100,1],
           [100,1]]
weights=np.array(weights)

model = SimpleModel(width=16, in_features=3, out_features=3)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X,Y) #works fine
model.fit(X,Y, sample_weight=weights) #also works fine

class SimpleLayer(tf.keras.layers.Layer):
    """Just dummy layer to illustrate sample_weight for layer"""
    def __init__(self, in_features, out_features, n):
        super(SimpleLayer, self).__init__()
        self.out_features = out_features
        self.in_features = in_features

        self.Gamma = self.add_weight(name='Gamma'+str(n),
                shape=(in_features, out_features), 
                initializer='glorot_normal', trainable=True)

    def call(self, inputs):
        #uses ragged map_flat_values for Ragged tensors to handle
        #variable number of jet
        xG = tf.ragged.map_flat_values(tf.matmul, inputs, self.Gamma)
        return xG

    
class SimpleModel(tf.keras.Model):
    """Composes SimpleLayer above to create simple network for ragged tensors"""
    def __init__(self, width, in_features, out_features, Sigma=tf.nn.leaky_relu):
        super(SimpleModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.width = width
        self.first_layer = SimpleLayer(self.in_features, self.width, 0)
        self.hidden = SimpleLayer(self.width, self.width, 1)
        self.last_layer = SimpleLayer(self.width, self.out_features, 2)
        self.Sigma = Sigma

    def call(self, inputs):
        #use map_flat_values to apply activation to ragged tensor
        x = tf.ragged.map_flat_values(self.Sigma, self.first_layer(inputs))
        x = tf.ragged.map_flat_values(self.Sigma, self.hidden(x))
        x = tf.ragged.map_flat_values(tf.nn.softmax, self.last_layer(x))
        return x