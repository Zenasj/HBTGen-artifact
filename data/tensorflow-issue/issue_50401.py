# tf.ragged.constant with shape (N_inputs, variable_length, N_input_channels), dtype=tf.float64
import tensorflow as tf
import numpy as np

class SimpleLayer(tf.keras.layers.Layer):
    """Just dummy layer to illustrate sample_weight for layer, supports RaggedTensor inputs."""
    def __init__(self, in_features, out_features, n):
        super(SimpleLayer, self).__init__()
        self.out_features = out_features
        self.in_features = in_features
        # Weight matrix: shape (in_features, out_features)
        self.Gamma = self.add_weight(
            name='Gamma'+str(n),
            shape=(in_features, out_features),
            initializer='glorot_normal',
            trainable=True)

    def call(self, inputs):
        # Use ragged.map_flat_values to handle RaggedTensor: apply matmul on flat_values
        xG = tf.ragged.map_flat_values(tf.matmul, inputs, self.Gamma)
        return xG

class SimpleModel(tf.keras.Model):
    """
    Composes SimpleLayer layers to create simple network for ragged tensor input.
    Applies nonlinearities and a softmax output per element.
    """
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
        # Apply first layer, then activation, all via ragged.map_flat_values to handle ragged inputs
        x = tf.ragged.map_flat_values(self.Sigma, self.first_layer(inputs))
        x = tf.ragged.map_flat_values(self.Sigma, self.hidden(x))
        x = tf.ragged.map_flat_values(tf.nn.softmax, self.last_layer(x))
        return x

class MyModel(tf.keras.Model):
    """
    Fused model encapsulating SimpleModel from the issue.
    This model supports ragged tensor inputs of shape (batch, variable_length, channels) as float64.
    """
    def __init__(self, width=16, in_features=3, out_features=3):
        super(MyModel, self).__init__()
        self.simple_model = SimpleModel(width, in_features, out_features)

    def call(self, inputs):
        # inputs expected as RaggedTensor, pass through simple_model
        return self.simple_model(inputs)

def my_model_function():
    """
    Returns an instance of MyModel with default initialization.
    """
    return MyModel()

def GetInput():
    """
    Returns a ragged Tensor input suitable for MyModel:
    - batch size 2
    - variable sequence length per batch: lengths 3 and 2
    - input channels: 3 (float64)
    State matches the example in the issue description.
    """
    X = [[[4.,3,2],[2,1,3],[-1,2,1]],
         [[1,2,3],[3,2,4]]]
    # RaggedTensor with ragged_rank=1 and dtype float64 to match the model expectations
    X_ragged = tf.ragged.constant(X, ragged_rank=1, dtype=tf.float64)
    return X_ragged

