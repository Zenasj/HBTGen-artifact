# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← MNIST grayscale images normalized [0,1]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture from the issue: Flatten + 9 dense ReLU layers + Dropout + final Dense softmax 10 classes
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense_layers = [
            tf.keras.layers.Dense(128, activation='relu', name=f'l_{i+1}th') for i in range(9)
        ]
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax', name='dense10')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)


def my_model_function():
    # Instantiate and compile the model to match the example code's setting
    model = MyModel()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model


def GetInput():
    # Return a random float tensor shaped (batch_size, 28, 28), matching MNIST-like input with values [0,1]
    batch_size = 64  # Chosen reasonable batch size for demonstration
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

