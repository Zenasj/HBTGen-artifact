# tf.random.uniform((B, 1), dtype=tf.float32)
import tensorflow as tf
import numpy as np
import threading

lock = threading.Lock()

class MyModel(tf.keras.Model):
    def __init__(self, a=1.0, b=0.0):
        super().__init__()
        # Parameters for linear regression y = a*x + b (for training data generation)
        self.a = a
        self.b = b
        # Build the model under a thread lock to avoid threading issues
        with lock:
            self.dense = tf.keras.layers.Dense(1, activation=None)
            # Create a dummy call to build weights
            self.build(input_shape=(None, 1))
            self.compile(loss='mean_squared_error',
                         optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                         metrics=['mse', 'mae', 'mape'])

    def call(self, inputs):
        return self.dense(inputs)

    def train_model(self, epochs=300, batch_size=10000):
        # Prepare training data: y = a*x + b with x in range 0 to 100000
        X_train = np.arange(100000).reshape(-1, 1).astype(np.float32)
        y_train = (self.a * X_train + self.b).astype(np.float32)
        X_test = (np.arange(100, 200) + 100000).reshape(-1, 1).astype(np.float32)
        y_test = (self.a * X_test + self.b).astype(np.float32)

        self.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                 validation_data=(X_test, y_test), verbose=0)

def my_model_function():
    # Instantiate MyModel with example a=5, b=6 as in user's sample configs
    return MyModel(a=5, b=6)

def GetInput():
    # Return a random input tensor suitable for MyModel:
    # Shape (batch_size, 1), float32 values in range 0..100
    # Batch size chosen as 4 for example convenience
    return tf.random.uniform((4, 1), dtype=tf.float32)

