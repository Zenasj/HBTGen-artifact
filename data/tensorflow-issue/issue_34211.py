# tf.random.uniform((B, 80), dtype=tf.int32) ← Input shape is (batch_size, sequence_length=80), integer tokens for embedding lookup

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer matching vocab size 20000 and embedding dim 128
        self.embedding = tf.keras.layers.Embedding(20000, 128)
        # LSTM layer with 128 units
        self.lstm = tf.keras.layers.LSTM(128)
        # Output dense layer with sigmoid activation
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)        # Shape: (B, 80, 128)
        x = self.lstm(x, training=training)  # LSTM output shape: (B, 128)
        return self.dense(x)              # Output shape: (B, 1), sigmoid score


def my_model_function():
    model = MyModel()
    # Compile with binary crossentropy loss and Adam optimizer 
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Generate a batch of random integer inputs within vocabulary range
    # Assuming batch size 32 as typical default
    batch_size = 32
    seq_length = 80
    # Use int32 tensor with values between 0 and 19999 (vocab size 20000)
    return tf.random.uniform(
        (batch_size, seq_length), minval=0, maxval=20000, dtype=tf.int32
    )

# ---
# ### Explanation / Assumptions
# - The reported issue stems from usage of LSTM inside a Sequential model saved/restored in SavedModel format, causing "No gradient defined for 'while'" errors.
# - The example Sequential model uses input sequences of variable length but padded to maxlen=80 at training.
# - Input shape: batch_size by 80 integers representing word indices.
# - Vocabulary size 20000, embedding dim 128, LSTM 128 units, Dense 1 sigmoid output.
# - The provided model matches exactly the structure from the issue:
#   ```
#   model = Sequential()
#   model.add(Embedding(20000, 128))
#   model.add(LSTM(128))
#   model.add(Dense(1, activation='sigmoid'))
#   ```
# - The `my_model_function()` produces a compiled instance ready for training.
# - `GetInput()` generates random integer token input matching the needed shape and dtype for the model.
# - This code does not replicate the saved/load problem or fix it but faithfully models the discussed architecture to be compatible with TF 2.20.0, and XLA compilation.
# - The gradient issue described was reportedly fixed in TF nightly versions after 2.0 (issue closed). This code is a minimal fully self-contained model matching the issue.
# - The input shape and dtypes are inferred from the usage in the IMDB dataset loading and pad_sequences with maxlen=80.
# This should serve as the canonical reference Python code extracted from the issue context, ready for further testing or debugging in TF 2.x environments.