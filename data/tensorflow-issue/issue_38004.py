# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← inferred input shape based on Fashion MNIST data

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple CNN-like architecture consistent with the original example
        # Using Sequential-like structure internally for clarity
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)
        self.softmax = tf.keras.layers.Softmax()
        
        # The original issue highlights a mismatch between 'accuracy' metric lookup and OneHot labels.
        # Here, the model forward outputs probabilities (softmax),
        # and the user should use an appropriate categorical accuracy metric to evaluate.
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense(x)
        x = self.softmax(x)
        return x

def my_model_function():
    # Instantiate MyModel and compile it with CategoricalCrossentropy and CategoricalAccuracy.
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")]
    )
    return model

def GetInput():
    # Generate a batch of input tensors with shape (batch_size, 28, 28) similar to Fashion MNIST grayscale images.
    # Use a batch size of 32 as a reasonable default.
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

