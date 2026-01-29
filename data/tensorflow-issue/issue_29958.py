# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST dataset (grayscale 28x28 images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple MNIST classifier model as described in the issue comments
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
        
        # Use a custom callback as a submodule (to demonstrate fusion and comparison)
        self.accuracy_callback = self.MyCallback()
        
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output
    
    class MyCallback(tf.keras.callbacks.Callback):
        """
        Callback that stops training once accuracy >= 99% and prints a message.
        Uses 'accuracy' key in logs dict to be compatible with TF 2.x versions.
        """
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            acc = logs.get('accuracy')
            if acc is not None and acc >= 0.99:
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True

def my_model_function():
    """
    Returns an instance of MyModel.
    """
    model = MyModel()
    # Compile with appropriate loss, optimizer, and metric to match the callback logs key 'accuracy'
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    """
    Returns a random tensor input with shape (batch_size, 28, 28) matching MNIST data format.
    The values are in [0,1), dtype float32.
    Assumed batch size of 32 for typical usage.
    """
    batch_size = 32
    # MNIST grayscale images: 28x28 single channel, normalized to [0,1]
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

