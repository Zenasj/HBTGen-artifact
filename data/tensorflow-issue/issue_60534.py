# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← input shape inferred from MNIST example (batch size unspecified)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple MNIST classifier matching the provided example
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)
        # Wrap softmax as part of a probability model
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        logits = self.dense2(x)
        probs = self.softmax(logits)
        return probs

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # It's typical TensorFlow practice to build the model by running on dummy input once
    dummy_input = GetInput()
    _ = model(dummy_input, training=False)
    return model

def GetInput():
    # Generate a random tensor input like MNIST images normalized to [0, 1]
    # Common batch size assumption: 1 for quick test
    # dtype=float32 to match typical TensorFlow input types
    # Note: Pixel values normalized to [0,1], shape (1, 28, 28)
    return tf.random.uniform(shape=(1, 28, 28), minval=0, maxval=1, dtype=tf.float32)

