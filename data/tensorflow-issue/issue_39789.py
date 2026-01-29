# tf.random.uniform((B, 2), dtype=tf.float32) ← inferred from the model input_dim=2 and input shape used
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original code uses a Sequential model with:
        # Dense(4, input_dim=2, activation='relu')
        # Dense(1, activation='linear')
        # We'll replicate that structure in this subclass.
        self.dense1 = tf.keras.layers.Dense(4, activation='relu', kernel_initializer='glorot_uniform', input_shape=(2,))
        self.dense2 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # The original training data was pairs like [0, 5], [1, 6], so the model input expects
    # shape (batch_size, 2), where the pair is (input_value, output_value).
    # However, the model input_dim=2 means the input to the model is 2 features,
    # presumably the first column (input value) only.
    # But in the original code model.fit was called on a (15,2) array, which caused confusion.

    # The original model definition suggests input_dim=2, so the input tensor should have shape (B, 2).

    # To keep consistent, we generate a batch of inputs shaped (15, 2).
    # We'll generate numbers 0 to 14 on axis 0 and their corresponding +5 values to mimic the training data:
    inputs = tf.constant([[float(i), float(i + 5)] for i in range(15)], dtype=tf.float32)
    return inputs

