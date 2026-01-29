# tf.random.uniform((B, 5), dtype=tf.float32), tf.random.uniform((B, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__(name='mymodel')
        # Two inner Dense layers with distinct names through assigning layers as attributes
        self.f = tf.keras.layers.Dense(units=10, name='dense_f')
        self.g = tf.keras.layers.Dense(units=10, name='dense_g')

    def build(self, input_shapes):
        # input_shapes expected as a tuple/list of 2 shapes: [(None, 5), (None, 3)]
        # Build layers with corresponding input shapes to create variables
        self.f.build(input_shapes[0])
        self.g.build(input_shapes[1])
        self.built = True

    def call(self, inputs):
        # inputs is expected to be a tuple (x, y)
        x, y = inputs
        return self.f(x) + self.g(y)

def my_model_function():
    # Instantiate the model and build it with example input shapes
    model = MyModel()
    model.build([(None, 5), (None, 3)])
    return model

def GetInput():
    # Return two tensors matching input shapes expected by MyModel:
    # First input: shape (batch_size, 5), second input: shape (batch_size, 3)
    # Use batch size = 2 as an example. dtype float32 is default for Dense.
    batch_size = 2
    input1 = tf.random.uniform((batch_size, 5), dtype=tf.float32)
    input2 = tf.random.uniform((batch_size, 3), dtype=tf.float32)
    return (input1, input2)

