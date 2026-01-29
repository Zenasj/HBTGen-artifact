# tf.random.uniform((2, 3), dtype=tf.float32)  ← inferred input shape from example: input shape (2, 3)

import tensorflow as tf
import tensorflow_addons as tfa

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original example used tf.keras.Sequential with two Dense layers 8 and 2 units
        self.dense1 = tf.keras.layers.Dense(8)
        self.dense2 = tf.keras.layers.Dense(2)
        
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Returns an instance of MyModel compiled with NovoGrad optimizer and sparse categorical crossentropy loss,
    # to reflect the original reported training setup where the issue appeared.
    model = MyModel()
    model.compile(
        optimizer=tfa.optimizers.NovoGrad(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    return model

def GetInput():
    # Returns a random tensor shaped (batch_size=2, features=3) matching input used in the example
    return tf.random.uniform(shape=(2, 3), dtype=tf.float32)

