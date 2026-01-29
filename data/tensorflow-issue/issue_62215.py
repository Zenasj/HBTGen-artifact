# tf.random.uniform((B, 1), dtype=tf.int32) ← input shape for Embedding layer is (batch_size, sequence_length=1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple model like the example: Embedding + Dense 1 unit sigmoid activation
        self.embedding = tf.keras.layers.Embedding(input_dim=100, output_dim=32)
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.embedding(inputs)
        # Embedding output shape: (B, 1, 32)
        # Flatten the last two dims or reduce sequence dimension (just take the embedding at position 0)
        # Because input shape is (B, 1), squeeze or remove middle dim before dense
        x = tf.squeeze(x, axis=1)  # shape (B, 32)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel with standard initialization
    model = MyModel()
    # Build the model by calling it on sample inputs (so variables are created)
    _ = model(tf.zeros((1, 1), dtype=tf.int32))
    # Compile with Adam experimental optimizer as used in the examples
    # Note: optimizer build will happen when using model.compile 
    model.compile(loss="binary_crossentropy", optimizer=tf.optimizers.experimental.Adam())
    return model

def GetInput():
    # Return random input tensor with shape (B, 1) matching the input shape expected by MyModel
    # Here batch size = 8 for demonstration purpose
    # dtype int32 because Embedding expects integer indices
    B = 8
    return tf.random.uniform((B, 1), minval=0, maxval=100, dtype=tf.int32)

