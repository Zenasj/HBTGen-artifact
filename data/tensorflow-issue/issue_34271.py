# tf.random.uniform((B, 8), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.custom_dense1 = Customized_DenseLayer(units=30, activation='relu')
        self.custom_dense2 = Customized_DenseLayer(units=1)

    def call(self, inputs):
        x = self.custom_dense1(inputs)
        return self.custom_dense2(x)

class Customized_DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(Customized_DenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            dtype=tf.float32,
            initializer='uniform',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            dtype=tf.float32,
            initializer='zeros',
            trainable=True)
        super(Customized_DenseLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch_size, features)
        # output: apply weights + bias then activation
        return self.activation(tf.matmul(inputs, self.kernel) + self.bias)

def my_model_function():
    # Return an instance of MyModel
    # No pretrained weights provided, just the model with initialized weights
    return MyModel()

def GetInput():
    # The model input expects a float32 tensor with shape (B, 8)
    # We'll generate a random uniform tensor with batch size 4 (arbitrary)
    batch_size = 4
    # Input features = 8 based on California housing data feature count
    input_tensor = tf.random.uniform((batch_size, 8), dtype=tf.float32)
    return input_tensor

