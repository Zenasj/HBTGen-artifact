# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Inferred input shape for the basic_classification model

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture based on basic_classification dataset example (Fashion MNIST)
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Instantiate and return the model.
    # This replicates the simple fashion MNIST model shown in the issue.
    model = MyModel()
    # Typically, the model would be compiled and trained externally.
    # Here we return the untrained model instance.
    return model

def GetInput():
    # Return a random tensor input matching the expected input shape for MyModel.
    # The input shape is (batch_size, 28, 28). We'll use batch size of 1 for example.
    # According to error messages and datasets, the dtype should be float32 (not float64)
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

