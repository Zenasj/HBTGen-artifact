# tf.random.uniform((32, 2), dtype=tf.float32) ← Inferred input shape based on FooEnv observation space shape (2,)
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The input shape inferred from the environment observations is (2,)
        # The model architecture is a simple MLP with two hidden layers and output layer for 5 actions
        self.flatten = Flatten(input_shape=(1, 2))  # window_length=1 from SequentialMemory, plus state shape (2,)
        self.dense1 = Dense(24, activation='relu')
        self.dense2 = Dense(24, activation='relu')
        self.output_layer = Dense(5, activation='linear')  # 5 actions

    def call(self, inputs):
        # inputs shape should be (batch_size, 1, 2), flatten to (batch_size, 2)
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output

def my_model_function():
    # Returns an instance of MyModel with initialized weights
    return MyModel()

def GetInput():
    # Produces a tensor with shape compatible with MyModel input: (batch_size, 1, 2)
    # Using batch size 32 as common batch size in original env/predict calls
    return tf.random.uniform((32, 1, 2), dtype=tf.float32)

