# tf.random.uniform((1, 224, 449, 3), dtype=tf.float32) ← input shape inferred from model.build and usage

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original bug example used tf.keras.Sequential with two Dense layers:
        #  - Dense(32, activation='relu')
        #  - Dense(5, activation='softmax')
        # Input shape is (1, 224, 449, 3), but Dense layers expect 2D input, so flatten first.
        # The workaround wraps the model with a reshape in call to flatten the input.

        # Base Sequential model as described
        self.base_model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),  # flatten (224,449,3) → (224*449*3,)
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])

    def call(self, x, training=False):
        # The wrapped model explicitly reshapes the input to [1,224,449,3]
        # We assume inputs come in that shape or possibly flattened as shape [1, 224*449*3],
        # so safeguard by reshaping to [1,224,449,3]
        x = tf.reshape(x, [1, 224, 449, 3])
        return self.base_model(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Return random uniform input matching the model's expected input shape [1, 224, 449, 3]
    # Use float32 dtype as per typical TensorFlow default
    return tf.random.uniform((1, 224, 449, 3), dtype=tf.float32)

