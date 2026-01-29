# tf.random.uniform((B, 784), dtype=tf.float32)  # Input shape inferred: batch size B, flattened 28x28 images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the original sequential model structure as per issue description
        self.dense1 = tf.keras.layers.Dense(512, activation='relu', input_shape=(784,))
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)  # logits output

        # Note: The model uses SparseCategoricalCrossentropy(from_logits=True) as loss,
        # so no activation on output layer.

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    
    model = MyModel()
    
    # Compile model exactly as in original example but with fixed metric:
    # As noted in the issue comments, 'accuracy' metric causes wrong metric after loading.
    # Instead, use 'sparse_categorical_accuracy' metric to avoid the bug.
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  # correct metric specification
    )
    return model

def GetInput():
    # Return a random tensor input matching model expectation: (batch_size, 784)
    # Using batch size 32 as example
    batch_size = 32
    # Values normalized similarly to MNIST: scaled between 0 and 1
    return tf.random.uniform((batch_size, 784), minval=0, maxval=1, dtype=tf.float32)

