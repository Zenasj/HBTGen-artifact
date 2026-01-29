# tf.random.uniform((10000, 32), dtype=tf.float32) ← inferred input shape from model input_shape=(32,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a simple fully-connected model matching the Keras Sequential example:
        # 3 Dense layers with relu or softmax activations and input shape (32,)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', name='dense_1')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu', name='dense_2')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax', name='dense_3')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def my_model_function():
    # Return an instance of MyModel; weights are randomly initialized
    return MyModel()

def GetInput():
    # Return random tensor input matching (batch_size=32, input_dim=32)
    # The batch size used in fit is 32, so use 32 here to match typical usage
    return tf.random.uniform((32, 32), dtype=tf.float32)

