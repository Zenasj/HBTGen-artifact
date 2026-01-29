# tf.random.uniform((B, 13), dtype=tf.float32)  # Input shape is (batch_size, 13) as per model input_shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreating the Sequential model described in the issue, 
        # converted to subclassing style with the same layers and activations
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(13,))
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(3, activation='relu')
        self.dense4 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)

def my_model_function():
    model = MyModel()
    # Compile with same loss, optimizer and metrics as in the example
    # Using SGD optimizer, binary_crossentropy loss, and accuracy metric, matching original code
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=["accuracy"])
    return model

def GetInput():
    # Input shape as per original model's input_shape=(13,)
    # Generating random float32 input tensor with batch size 4 (arbitrary small batch size)
    return tf.random.uniform((4, 13), dtype=tf.float32)

