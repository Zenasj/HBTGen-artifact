# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape inferred as (batch_size, 784) flattened MNIST images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(784,))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(10)  # logits output

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel with compiled optimizer, loss, and metric compatible with sparse integer labels.
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

def GetInput():
    # Return a batch of random inputs with shape (batch_size, 784), dtype float32 to simulate MNIST flattened images
    batch_size = 32  # common batch size
    # Random float32 tensor with values in [0,1)
    return tf.random.uniform((batch_size, 784), dtype=tf.float32)

