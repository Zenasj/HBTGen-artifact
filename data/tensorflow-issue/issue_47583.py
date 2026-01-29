# tf.random.uniform((100, 1), dtype=tf.float32) ← Input shape inferred from dataset_size and model input shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # A simple single Dense layer model as per the example
        self.dense = tf.keras.layers.Dense(1, input_shape=(1,))

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Instantiate and compile the model 
    model = MyModel()
    model.compile(optimizer='sgd', loss='mse')
    return model

def GetInput():
    # Generate a random input tensor matching the input shape (batch_size, 1)
    # Using batch size = 100 (dataset_size) to mirror the original dataset shape.
    return tf.random.uniform((100, 1), dtype=tf.float32)

