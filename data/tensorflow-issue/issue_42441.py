# tf.random.uniform((4, 2), dtype=tf.float32) ← inferred input shape from example x = np.array([[0,0],[0,1],[1,0],[1,1]])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple sequential-like model with two dense layers as per the original example
        self.dense1 = tf.keras.layers.Dense(units=2)
        self.dense2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input shape (4 samples, 2 features)
    # Using uniform distribution over [0,1) as analogue to binary 0/1 in example
    return tf.random.uniform((4,2), dtype=tf.float32)

