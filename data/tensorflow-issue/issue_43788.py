# tf.random.uniform((B,)) ← Input is a 1D float tensor with shape equal to features_num (assumed 10 here)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, features_num=10, nr_classes=3):
        super().__init__()
        # The original model is a simple MLP with:
        # Flatten input -> Dense 128 ReLU -> Dense nr_classes softmax
        # Here we assume features_num and nr_classes from typical setup, adjust as needed.
        self.flatten = tf.keras.layers.Flatten(input_shape=(features_num,))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(nr_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default parameters
    # Adjust features_num and nr_classes as per your dataset/model
    return MyModel(features_num=10, nr_classes=3)

def GetInput():
    # Return a random tensor input matching the input expected by the model
    # Here, we assume input feature vector of size 10 (features_num)
    # Batch size is 1 for demonstration
    return tf.random.uniform((1, 10), dtype=tf.float32)

