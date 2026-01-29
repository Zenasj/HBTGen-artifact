# tf.random.uniform((12, 100, 20), dtype=tf.float32) ← input shape inferred from batch size 12, sequence length 100, features 20
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreate the model from the issue inside a keras.Model subclass
        self.lstm = tf.keras.layers.LSTM(64)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(3, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel.
    # In the issue, the model is compiled with sparse_categorical_crossentropy,
    # Adam, and accuracy metrics, but here we only construct the model.
    # Compilation can be done externally if needed.
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model input shape used.
    # The original dataset has batches of size 12, sequences of length 100, 20 features.
    # Use tf.random.uniform for float32 inputs.
    batch_size = 12
    seq_length = 100
    features = 20
    return tf.random.uniform((batch_size, seq_length, features), dtype=tf.float32)

