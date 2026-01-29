# tf.random.uniform((B,), dtype=tf.float32) ← Based on the training input shape (10000,) used in the example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, n_layers=5, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.n_layers = n_layers
        # Create layers stored in a dict like in the example
        self.map = {}
        for i in range(self.n_layers):
            self.map[str(i)] = tf.keras.layers.Dense(units=1)

    def call(self, X, training=False):
        # Sequentially apply dense layers
        for i in range(self.n_layers):
            X = self.map[str(i)](X)
        return X

def my_model_function():
    # Return an instance of MyModel initialized with default 5 layers, as per example
    return MyModel(n_layers=5)

def GetInput():
    # Return a random tensor shaped (10000,), dtype float32, matching the example's input to fit
    # This matches the np.ones shape used for x in the example fit calls
    # Shape is (batch_size,) = (10000,)
    return tf.random.uniform(shape=(10000,), dtype=tf.float32)

