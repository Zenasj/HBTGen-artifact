# tf.random.uniform((B, 100, 1), dtype=tf.float32) ← Inferred input shape based on data preprocessing and model input shape

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dropout, Dense
from tensorflow.keras import Model, Input

class RandActivation(tf.keras.layers.Layer):
    """
    Custom random activation layer that outputs 1 if random number > input,
    else outputs 0. Designed to operate element-wise or with single neuron input.
    
    In original code intent:
    For each input x, generate random uniform scalar r,
    output 1 if r > x else 0.
    """
    def __init__(self, **kwargs):
        super(RandActivation, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        # Inputs expected shape: (batch, 1)
        # We'll generate random floats in [0,1) of shape (batch, 1)
        # Compare element-wise: output 1 if random > input else 0.

        # To ensure the operation is compatible with TF graph and jit_compile,
        # use tf.random.uniform and tf.where without TF Variables.

        random_vals = tf.random.uniform(shape=tf.shape(inputs), dtype=inputs.dtype)
        outputs = tf.where(random_vals > inputs, tf.ones_like(inputs), tf.zeros_like(inputs))
        return outputs

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Following the architecture described:
        # Conv1D(150, kernel_size=50, activation='relu', input_shape=(100,1))
        # MaxPooling1D(pool_size=4)
        # GRU(200)
        # Dropout(0.25)
        # Dense(128, relu)
        # Dense(35, relu)
        # Dropout(0.25)
        # Dense(1, relu, use_bias=False)
        # Dense(1, custom random activation layer, use_bias=False)

        self.conv1d = Conv1D(150, kernel_size=50, activation='relu')
        self.maxpool = MaxPooling1D(pool_size=4)
        self.gru = GRU(200)
        self.dropout1 = Dropout(0.25)
        self.dense128 = Dense(128, activation='relu')
        self.dense35 = Dense(35, activation='relu')
        self.dropout2 = Dropout(0.25)
        self.dense_relu = Dense(1, activation='relu', use_bias=False)
        self.rand_activation = RandActivation()

    def call(self, inputs, training=None):
        x = self.conv1d(inputs)
        x = self.maxpool(x)
        x = self.gru(x)
        if training:
            x = self.dropout1(x, training=training)
        else:
            x = self.dropout1(x, training=False)
        x = self.dense128(x)
        x = self.dense35(x)
        if training:
            x = self.dropout2(x, training=training)
        else:
            x = self.dropout2(x, training=False)
        x = self.dense_relu(x)
        # reshape to (batch, 1) if needed
        # The dense_relu outputs shape (batch, 1)
        # rand_activation layer expects same shape
        x = tf.reshape(x, (-1, 1))
        out = self.rand_activation(x)
        return out

def my_model_function():
    """
    Returns an instance of MyModel, constructed and compiled with
    mean_squared_error loss and Adam optimizer with lr=0.0001, accuracy metric.
    """
    model = MyModel()
    # Compile with similar setup as original:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mean_squared_error',
        metrics=['accuracy']
    )
    return model

def GetInput():
    """
    Returns a random tensor matching the expected input shape to MyModel.
    From the original code:
    input_shape = (100,1), batch size inferred to be arbitrary - use batch=32
    """
    batch_size = 32
    input_shape = (100, 1)
    return tf.random.uniform((batch_size,) + input_shape, dtype=tf.float32)

