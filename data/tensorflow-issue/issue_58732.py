# tf.random.uniform((B, 64), dtype=tf.float32), tf.random.uniform((B, 16), dtype=tf.float32), tf.random.uniform((B, 16), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers corresponding to original model
        self.layer0 = tf.keras.layers.Dense(3, activation="relu", name="layer0")
        self.layer1 = tf.keras.layers.Dense(5, activation="relu", name="layer1")
        self.layer2 = tf.keras.layers.Dense(7, activation="relu", name="layer2")
        self.output0_dense = tf.keras.layers.Dense(3, name="output0")
        self.output1_dense = tf.keras.layers.Dense(7, name="output1")

    def call(self, inputs):
        # Expect inputs as a list or tuple of three tensors
        input0, input1, input2 = inputs
        # Forward pass through each dense layer branch
        x0 = self.layer0(input0)
        x1 = self.layer1(input1)
        x2 = self.layer2(input2)
        # Concatenate for output0 and output1
        concat0 = tf.concat([x0, x1, x2], axis=-1)
        concat1 = tf.concat([x0, x1, x2], axis=-1)
        out0 = self.output0_dense(concat0)
        out1 = self.output1_dense(concat1)
        return [out0, out1]

def my_model_function():
    # Instantiate and return the model with correct layers and naming
    model = MyModel()
    # Compile with optimizer and loss to match original usage (SGD and MSE)
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model

def GetInput():
    # Create random inputs matching the input shapes specified:
    #   input0 shape (B,64), input1 shape (B,16), input2 shape (B,16)
    # Using batch size B=1 for example
    batch_size = 1
    input0 = tf.random.uniform((batch_size, 64), dtype=tf.float32)
    input1 = tf.random.uniform((batch_size, 16), dtype=tf.float32)
    input2 = tf.random.uniform((batch_size, 16), dtype=tf.float32)
    return [input0, input1, input2]

