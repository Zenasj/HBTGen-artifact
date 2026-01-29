# tf.random.uniform((B, 128), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers mirroring the functional model described
        self.dense_1 = tf.keras.layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = tf.keras.layers.Dense(64, activation='relu', name='dense_2')
        self.dense_3 = tf.keras.layers.Dense(
            10, activation=None, name='pre_softmax')  # pre-softmax logits layer
        
        # Lambda layer to print intermediate tensor before softmax
        self.print_layer = tf.keras.layers.Lambda(self.log_pre_softmax)
        
        # Final softmax activation layer
        self.softmax = tf.keras.layers.Activation('softmax', name='predictions')

    def log_pre_softmax(self, x):
        # Print tensor min/max statistics and check for NaNs as an example of debugging inside the model
        # This matches the recommended approach from the issue to use tf.print inside a Lambda layer
        tf.print("Pre softmax, min:", tf.reduce_min(x), "max:", tf.reduce_max(x))
        tf.debugging.check_numerics(x, "NaN detected in pre-softmax layer")
        return x

    def call(self, inputs, training=False):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.print_layer(x)
        output = self.softmax(x)
        return output

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor with shape (batch_size=32, 128), matching input shape for the model
    # Using float32 as typical input dtype for TF models
    return tf.random.uniform((32, 128), dtype=tf.float32)

