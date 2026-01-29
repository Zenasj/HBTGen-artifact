# tf.random.uniform((400,)) ← Input shape inferred from model Input layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the reported Sequential model layers here as Dense layers with relu and sigmoid activations.
        self.dense1 = tf.keras.layers.Dense(400, activation='relu')
        self.dense2 = tf.keras.layers.Dense(400, activation='relu')
        self.dense3 = tf.keras.layers.Dense(400, activation='relu')
        self.dense4 = tf.keras.layers.Dense(400, activation='relu')
        self.dense5 = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Weights are not preloaded; random initialization consistent with the example
    return model

def GetInput():
    # Return a random tensor matching the input shape expected by MyModel.
    # Batch size is arbitrary; 32 chosen here as a reasonable default for a small batch.
    return tf.random.uniform((32, 400), dtype=tf.float32)

