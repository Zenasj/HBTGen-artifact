# tf.random.uniform((BATCH_SIZE, 5), dtype=tf.float32) ← inferred input shape from provided dataset (num_features=5)
import tensorflow as tf

BATCH_SIZE = 4  # as given in original code

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self._layer = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        # The original code inspects keras.backend.learning_phase(),
        # which is a symbolic tensor related to training phase.
        # In TF2, learning_phase() is deprecated in behavior, but we replicate original intent.

        # Get learning_phase tensor (likely 0d scalar in TF2 eager, but originally problematic)
        learning_phase = tf.keras.backend.learning_phase()

        # We'll try to illustrate the core issue in original code: 
        # learning_phase shape and value behavior, but since TF2 disables tf.Print,
        # replace with tf.print (for side effects).
        # Since side effects in tf.function are tricky, we just replicate the key logic here.

        # Show dimension info (symbolic in TF1, here is static)
        # learning_phase shape may be None or 0d
        # We replicate prints using tf.print for demonstration:
        tf.print("learning_phase shape.ndims:", tf.shape(learning_phase))
        tf.print("learning_phase value:", learning_phase)
        tf.print("reduce_any(learning_phase):", tf.reduce_any(learning_phase))

        x = self._layer(inputs)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random float32 input tensor of shape (BATCH_SIZE, 5)
    # consistent with get_dataset() input shape from issue
    return tf.random.uniform((BATCH_SIZE, 5), dtype=tf.float32)

