# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← assuming input batch B, grayscale 28x28 images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the model architecture described:
        # Input shape: (28, 28)
        # Reshape to (28, 28, 1)
        self.reshape = tf.keras.layers.Reshape(target_shape=(28, 28, 1))
        self.conv = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)  # logits for 10 classes
    
    def call(self, inputs, training=False):
        x = self.reshape(inputs)         # (B, 28, 28) -> (B, 28, 28, 1)
        x = self.conv(x)                 # conv layer with relu
        x = self.pool(x)                 # max pooling
        x = self.flatten(x)              # flatten for dense layer
        logits = self.dense(x)           # final dense layer without activation
        return logits

def my_model_function():
    model = MyModel()
    # Here, typically one would load weights or compile, but the prompt only requires model instance
    # The original uses sparse categorical crossentropy with logits=True
    # For completeness, compiling with similar settings:
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # The model expects input shape (B, 28, 28), float32 normalized [0,1]
    # We generate a batch size of 1 for simplicity
    batch_size = 1
    # Using uniform random inputs in range [0,1], float32
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

