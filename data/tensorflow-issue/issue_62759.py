# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape inferred from MNIST flattened 28x28 images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple 3-layer dense network matching the example in the issue
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.out_layer = tf.keras.layers.Dense(10)  # logits for 10 classes

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out_layer(x)

def my_model_function():
    # Instantiate MyModel and compile it to mimic the example setup
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

def GetInput():
    # Return a batch of random inputs shaped like the MNIST flattened images (batch size 32)
    batch_size = 32
    input_shape = (batch_size, 784)
    return tf.random.uniform(input_shape, dtype=tf.float32)

