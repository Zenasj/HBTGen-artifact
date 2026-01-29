# tf.random.uniform((B, seq_length), dtype=tf.int32) ← Input is token IDs sequences for embedding lookup

import tensorflow as tf
import numpy as np

class Attention_Model(tf.keras.layers.Layer):
    def __init__(self, seq_length, units):
        super().__init__()
        self.seq_length = seq_length
        self.units = units
        self.lstm = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(units=seq_length)
        self.softmax = tf.keras.layers.Softmax(axis=1)
        self.repeat_vector = tf.keras.layers.RepeatVector(seq_length)
    
    def call(self, X):
        # X shape: (batch_size, seq_length, feature_dim)
        batch_size = tf.shape(X)[0]

        # Initialize output tensor (batch, seq_length, units)
        # Use a TensorArray to accumulate results over time steps (seq_length)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=self.seq_length, dynamic_size=False)

        # Initial state s_0 as zeros (batch, units)
        s = tf.zeros(shape=(batch_size, self.units), dtype=tf.float32)

        for i in tf.range(self.seq_length):
            # Repeat s to (batch, seq_length, units) - note: self.repeat_vector repeats to fixed length
            s_repeated = tf.repeat(tf.expand_dims(s, axis=1), repeats=self.seq_length, axis=1)  # shape: (batch, seq_length, units)

            # Concatenate along feature dim: X (batch, seq_length, feature_dim) + s_repeated (batch, seq_length, units)
            concat_X = tf.concat([X, s_repeated], axis=-1)  # shape: (batch, seq_length, feature_dim + units)
            
            # Calculate alphas (attention weights) over sequence dimension
            alphas = self.softmax(self.dense(concat_X))  # shape: (batch, seq_length, seq_length)

            # Extract the i-th timestep's attention weights for all batches: shape (batch, seq_length)
            alpha_i = alphas[:, :, i]

            # Expand dims to broadcast for multiplication: (batch, seq_length, 1)
            alpha_i_exp = tf.expand_dims(alpha_i, axis=-1)

            # Compute attention-weighted sum: sum over seq_length dimension
            weighted_sum = tf.reduce_sum(X * alpha_i_exp, axis=1, keepdims=True)  # (batch, 1, feature_dim)

            # Pass weighted sum through LSTM step
            # Because LSTM expects 3D input: (batch, timesteps, features)
            # We'll pass one time-step input to get output for that timestep
            lstm_out, h_state, c_state = self.lstm(weighted_sum, initial_state=None)

            # lstm_out shape: (batch, 1, units), take first timestep output for final output
            output_i = lstm_out[:, 0, :]  # (batch, units)

            outputs_ta = outputs_ta.write(i, output_i)

            # Update s for next iteration to the cell output h_state
            s = h_state

        # Stack outputs: (seq_length, batch, units) -> transpose to (batch, seq_length, units)
        outputs = outputs_ta.stack()
        outputs = tf.transpose(outputs, perm=[1, 0, 2])

        return outputs


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=10000, embed_dim=100, seq_length=21, classes=5):
        """
        Construct a model with:
         - Embedding layer
         - Two Bidirectional LSTMs
         - Attention model layer (custom)
         - Several Dense layers with Dropout and sigmoid activations
         - Final Dense softmax output layer with 'classes' units

        Defaults and sizes are assumed based on issue content.
        """
        super().__init__()

        self.seq_length = seq_length
        self.classes = classes

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="embd")

        self.lstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=100, return_sequences=True, name="lstm1"),
            name="bd1"
        )
        self.lstm2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=100, return_sequences=True, name="lstm2"),
            name="bd2"
        )

        self.attention_layer = Attention_Model(seq_length=seq_length, units=200)

        self.dense1 = tf.keras.layers.Dense(units=80, kernel_regularizer=tf.keras.regularizers.l2(), name="dense1")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.5)
        self.act1 = tf.keras.layers.Activation('sigmoid')

        self.dense2 = tf.keras.layers.Dense(units=50, kernel_regularizer=tf.keras.regularizers.l2(), name="dense2")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.4)
        self.act2 = tf.keras.layers.Activation('sigmoid')

        self.dense3 = tf.keras.layers.Dense(units=30, kernel_regularizer=tf.keras.regularizers.l2(), name="dense3")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.3)
        self.act3 = tf.keras.layers.Activation('sigmoid')

        self.dense4 = tf.keras.layers.Dense(units=classes, name="dense4")
        self.dropout4 = tf.keras.layers.Dropout(rate=0.2)
        self.output_activation = tf.keras.layers.Activation('softmax')

    def call(self, inputs, training=False):
        """
        inputs: tf.Tensor shape (batch, seq_length) containing token indices
        
        Returns:
          tf.Tensor shape (batch, classes) with softmax class probabilities
        """
        # Embedding lookup (batch, seq_length, embed_dim)
        x = self.embedding(inputs)

        # Bidirectional LSTMs
        x = self.lstm1(x)
        x = self.lstm2(x)

        # Attention model outputs (batch, seq_length, units=200)
        x = self.attention_layer(x)

        # Flatten sequence dimension for dense layers, or apply dense layers time-distributed.
        # From context it seems dense layers apply to each time step, but to produce final classification,
        # we reduce to sequence representation - here we take the last timestep for classification
        # Alternatively, average pooling over timesteps

        # We'll do GlobalAveragePooling1D over sequence axis to get (batch, units)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Fully connected layers with dropout and sigmoid activations
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.act1(x)

        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.act2(x)

        x = self.dense3(x)
        x = self.dropout3(x, training=training)
        x = self.act3(x)

        x = self.dense4(x)
        x = self.dropout4(x, training=training)
        output = self.output_activation(x)

        return output


def my_model_function():
    """
    Instantiate MyModel with likely parameters:
    - vocab_size: 10000 (typical, inferred)
    - embed_dim: 100 (from embedding output_dim)
    - seq_length: 21 (from Attention_Model seq_length)
    - classes: 5 (assumed label classes from example)
    """
    return MyModel(vocab_size=10000, embed_dim=100, seq_length=21, classes=5)


def GetInput():
    """
    Generate a random input tensor matching expected input shape to MyModel:
    (batch_size, seq_length) with integer token IDs in [0, vocab_size-1]

    Choosing batch_size=4 as reasonable default.
    """
    batch_size = 4
    seq_length = 21
    vocab_size = 10000
    input_tensor = tf.random.uniform(
        shape=(batch_size, seq_length),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )
    return input_tensor

