# tf.random.uniform((24, 1), dtype=tf.float32) ← Input shape inferred from batch size and dataset preparation in the issue (BATCH_SIZE=24, shape=(1,))

import tensorflow as tf
from tensorflow.keras import layers, Input, optimizers, losses
from tensorflow.keras import mixed_precision

# Set mixed precision policy as in the example code
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the model architecture from the issue:
        # input shape = (1,)
        # Four Dense layers with 256 units + leaky_relu activation (using tf.keras.layers.LeakyReLU explicitly,
        # since 'leaky_relu' is not a built-in string activation).
        self.dense1 = layers.Dense(256)
        self.act1 = layers.LeakyReLU()
        self.dense2 = layers.Dense(256)
        self.act2 = layers.LeakyReLU()
        self.dense3 = layers.Dense(256)
        self.act3 = layers.LeakyReLU()
        self.dense4 = layers.Dense(256)
        self.act4 = layers.LeakyReLU()
        self.out = layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        x = self.dense3(x)
        x = self.act3(x)
        x = self.dense4(x)
        x = self.act4(x)
        x = self.out(x)
        return x

def my_model_function():
    # Create and compile the model, reflecting the original setup
    
    model = MyModel()
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.01),
        loss=losses.MeanAbsoluteError(),
        jit_compile=True  # Important, as per the issue
    )
    return model

def GetInput():
    # Returns a random tensor simulating a batch of size 24 with input shape (1,)
    # dtype float32 to be compatible with mixed precision policy (inputs usually float32),
    # note that the policy affects model layers, not input dtypes directly.
    # This matches the dataset pipeline in the issue where input shape was (1,)
    # and batch size was 24.
    return tf.random.uniform((24, 1), dtype=tf.float32)

