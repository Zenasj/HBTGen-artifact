# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← MNIST images batch of grayscale 28x28

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the MNIST CNN model from the reported code in the issue
        filters = 48
        kernel_size = 7
        units = 24
        
        self.conv = tf.keras.layers.Conv2D(filters=filters, 
                                           kernel_size=(kernel_size, kernel_size), 
                                           activation='relu', 
                                           input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def my_model_function():
    # Instantiate model and compile with same parameters as given
    model = MyModel()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Per the model, input shape is 28x28 grayscale images with batch dimension unspecified
    # Here we provide a batch size of 32 as used in the original example
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)

