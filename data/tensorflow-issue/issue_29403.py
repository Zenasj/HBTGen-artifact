# tf.random.uniform((B, 10), dtype=tf.float32) ← input shape inferred from input_shape=(10,) in Flatten

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreate the model architecture described in the issue
        # A Flatten layer on input_shape (10,) is effectively a no-op, 
        # but included as per original model for replication
        self.flatten = tf.keras.layers.Flatten(input_shape=(10,))
        self.dense1 = tf.keras.layers.Dense(100)
        self.elu1 = tf.keras.layers.ELU()
        self.dense2 = tf.keras.layers.Dense(500)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.elu2 = tf.keras.layers.ELU()
        self.dense3 = tf.keras.layers.Dense(2, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.elu1(x)
        x = self.dense2(x)
        x = self.batch_norm(x, training=training)
        x = self.elu2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Instantiate the model and compile it with the same settings as in original code.
    model = MyModel()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )
    # Fit with dummy data to initialize weights as done in original issue snippet.
    import numpy as np
    x_dummy = np.ndarray((5000, 10), dtype=float)
    y_dummy = np.array([0, 1] * 2500)
    model.fit(x_dummy, y_dummy, epochs=1, verbose=0)
    return model

def GetInput():
    # Provide a random input tensor of shape (1, 10), dtype float32 as expected by the model.
    return tf.random.uniform((1, 10), dtype=tf.float32)

