# tf.random.uniform((B, 250, ), dtype=tf.float32) ← Input shape inferred from train_dataset.keys() count (250 features)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Build the same architecture as in the original Sequential model:
        # 3 Dense layers with relu activations, then a final linear output
        self.dense1 = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(512, activation='relu')
        self.dense3 = layers.Dense(256, activation='relu')
        self.out_layer = layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        output = self.out_layer(x)
        return output

def my_model_function():
    # Return a new instance of MyModel.
    # Weights can be loaded later if needed.
    return MyModel()

def GetInput():
    # Based on the dataset columns in the issue, there are 250 input features (excluding label)
    # Let's create a single batch input with shape (1, 250)
    # Use float32 as dtype to match typical Keras layers input dtype
    import tensorflow as tf
    B = 1       # batch size 1
    feature_dim = 250  # inferred number of input features from train_dataset.keys()

    # To simulate normalized input as in the original code (which did no normalization),
    # use uniform values in range [0, 1).
    return tf.random.uniform((B, feature_dim), dtype=tf.float32)

