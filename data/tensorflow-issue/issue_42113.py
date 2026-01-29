# tf.random.uniform((B, 64, 2048), dtype=tf.float32) ← input shape inferred from encoder input shape and example code

import tensorflow as tf

# Attention mechanism using Bahdanau style attention
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features shape: (batch_size, 64, embedding_dim)
        # hidden shape: (batch_size, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # (batch_size, 1, hidden_size)

        # Score calculation (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # Attention weights (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # Context vector as weighted sum (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# Encoder to process extracted CNN features (pre-extracted from something like InceptionV3)
class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # Fully connected layer to transform features to embedding dim
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

# Decoder RNN with attention, outputs vocabulary logits
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # Attention and context vector
        context_vector, attention_weights = self.attention(features, hidden)

        # Embed input word (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # Concatenate context vector and embedding (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Pass through GRU
        output, state = self.gru(x)

        # Fully connected layers
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))  # Flatten for final dense layer

        x = self.fc2(x)  # Output logits (batch_size * seq_len, vocab_size)

        return x, state, attention_weights

    def reset_states(self, batch_size):
        return tf.zeros((batch_size, self.units))


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # These must be defined with assumed hyperparameters:
        # These parameters should align with the placeholders used in the training code snippet.
        self.embedding_dim = 256
        self.units = 512

        # Note: vocab_size and top_k are not known, we choose a placeholder size.
        # For example, vocab_size = 5000 (You can adjust as needed)
        self.vocab_size = 5000 

        self.encoder = CNN_Encoder(self.embedding_dim)
        self.decoder = RNN_Decoder(self.embedding_dim, self.units, self.vocab_size)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        """
        Expects inputs to be a tuple: (img_tensor, target)
        img_tensor: (batch_size, 64, 2048)  extracted image features
        target: (batch_size, max_seq_len)  integer word indices for target caption sequence
        """
        img_tensor, target = inputs
        batch_size = tf.shape(target)[0]

        hidden = self.decoder.reset_states(batch_size=batch_size)

        dec_input = tf.expand_dims([1] * batch_size, 1)  # Assuming 1 is <start> token index

        # Encode images
        features = self.encoder(img_tensor)

        outputs = []
        for i in range(1, tf.shape(target)[1]):
            # Forward pass through decoder for each time step
            predictions, hidden, _ = self.decoder(dec_input, features, hidden)
            outputs.append(predictions)
            # Teacher forcing - use true target next word as next input
            dec_input = tf.expand_dims(target[:, i], 1)

        # Stack outputs for all time steps: list of (batch_size * 1, vocab_size) tensors
        # We reshape to (batch_size, seq_len - 1, vocab_size)
        stacked_outputs = tf.stack(outputs, axis=1)
        output_shape = tf.concat([[batch_size], [tf.shape(target)[1] - 1], [self.vocab_size]], axis=0)
        stacked_outputs = tf.reshape(stacked_outputs, output_shape)

        return stacked_outputs


def my_model_function():
    # Create and return a fresh MyModel instance
    return MyModel()


def GetInput():
    # Produce dummy input matching the model's expected input
    # img_tensor shape: (batch_size, 64, 2048), dtype float32
    # target shape: (batch_size, max_seq_len), dtype int32 (word indices)

    batch_size = 2  # example batch size
    seq_len = 10  # example caption length

    img_tensor = tf.random.uniform((batch_size, 64, 2048), dtype=tf.float32)
    target = tf.random.uniform((batch_size, seq_len), minval=2, maxval=4999, dtype=tf.int32)  
    # minval=2 to avoid <start> token index=1, but random is fine as placeholder

    # Forcing the first target token to <start> token with index 1 (matching training code)
    target = tf.concat([tf.ones((batch_size, 1), dtype=tf.int32), target[:, 1:]], axis=1)

    return (img_tensor, target)

