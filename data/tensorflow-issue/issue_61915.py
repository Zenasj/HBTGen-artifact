# tf.random.uniform((B, 2, 3), dtype=tf.float32) ← Input shape inferred from keras.Input(shape=(2,3))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers used in the original model snippet
        self.dense1 = tf.keras.layers.Dense(512)
        # Note: The original output layer used 'softmax' activation with 1 unit, which is unusual.
        # Softmax usually implies multi-class classification. We'll keep as is to match original code.
        self.output_layer = tf.keras.layers.Dense(1, activation='softmax', name='prediction')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.output_layer(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching input shape of (batch_size, 2, 3)
    # Here batch size is arbitrary, let's pick 4 for example
    batch_size = 4
    return tf.random.uniform((batch_size, 2, 3), dtype=tf.float32)

