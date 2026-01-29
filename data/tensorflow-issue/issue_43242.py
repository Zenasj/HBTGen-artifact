# tf.random.uniform((1, 6, 3), dtype=tf.float32) ← inferred input shape from the original model input batch_input_shape=(1,6,3)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten input of shape (1, 6, 3) -> (1, 18)
        self.flatten = tf.keras.layers.Flatten()
        # Dense with 1 neuron, ReLU activation as per user's original model
        self.dense = tf.keras.layers.Dense(1, activation='relu')
        
    def call(self, inputs):
        x = self.flatten(inputs)
        output = self.dense(x)
        return output

def my_model_function():
    # Returns an instance of MyModel - weights are randomly initialized as in the original model (no saved weights provided)
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected input: batch size = 1, time steps = 6, features =3, dtype float32
    return tf.random.uniform((1, 6, 3), dtype=tf.float32)

