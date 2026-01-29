# tf.random.uniform((1, 1), dtype=tf.int64), tf.random.uniform((1, 64, 256), dtype=tf.float32), tf.random.uniform((1, 512), dtype=tf.float32)
import tensorflow as tf

# Placeholder attention mechanism as per BahdanauAttention usage in decoder
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features shape: (batch_size, 64, embedding_dim)
        # hidden shape: (batch_size, units)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # (batch_size, 1, units)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))  # (batch_size, 64, units)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)  # (batch_size, 64, 1)

        context_vector = attention_weights * features  # (batch_size, 64, embedding_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size, embedding_dim)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim, features_shape=2048):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)
        self.features_shape = features_shape
        self.embedding_dim = embedding_dim

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 64, 2048), dtype=tf.float32)])
    def call(self, x):
        # x has shape (batch_size=1, 64, features_shape)
        x = self.fc(x)  # shape (1, 64, embedding_dim)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       unroll=True)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    # Adjusted input_signature to expect tf.int64 for x input to match inference input dtype,
    # features and hidden are float32 as per encoder and decoder states.
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 1], dtype=tf.int64),
        tf.TensorSpec(shape=[1, 64, 256], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 512], dtype=tf.float32)])
    def call(self, x, features, hidden):
        # Attention context and weights
        context_vector, attention_weights = self.attention(features, hidden)

        # Embedding x (batch_size=1, 1) -> (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # Concatenate context vector with embedded input (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Pass concatenated tensor through GRU
        output, state = self.gru(x)  # output shape: (batch_size, 1, units)

        # Pass thru FC layers
        x = self.fc1(output)  # shape: (batch_size, 1, units)

        # Flatten for final dense layer
        x = tf.reshape(x, (-1, x.shape[2]))  # shape: (batch_size * 1, units)

        x = self.fc2(x)  # shape: (batch_size * 1, vocab_size)

        return x, state, attention_weights

    def reset_states(self, batch_size):
        return tf.zeros((batch_size, self.units), dtype=tf.float32)


class MyModel(tf.keras.Model):
    def __init__(self, embedding_dim=256, units=512, vocab_size=5000, features_shape=2048):
        """
        Combined image captioning model with Encoder and Decoder.
        For simplicity, vocab_size is assumed, embedding_dim and units as typical values.
        """
        super(MyModel, self).__init__()
        self.encoder = CNN_Encoder(embedding_dim=embedding_dim, features_shape=features_shape)
        self.decoder = RNN_Decoder(embedding_dim=embedding_dim, units=units, vocab_size=vocab_size)
        self.max_length = 20  # Example max caption length

        # We fix some tokens for demonstration; in practice these come from tokenizer.
        self.start_token_id = 1  # Typical token id for <start>
        self.end_token_id = 2    # Typical token id for <end>

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 299, 299, 3), dtype=tf.float32)])
    def call(self, images):
        """
        Forward pass that mimics the 'evaluate' function from the issue:
         - images: batch size 1, typical image tensor of shape (1, H, W, C)
         - outputs: predicted token ids sequence tensor of shape (max_length,)
        
        Assumptions:
          - image_features_extract_model is integrated inside this function to get features_shape=2048.
          - For simplicity, we replace image_features_extract_model with a dummy conv model here.
        """
        batch_size = 1

        # Placeholder image features extractor, emulating image_features_extract_model:
        # Normally, this would be InceptionV3 or similar truncated model outputting shape (1, 8, 8, 2048).
        # We'll simulate with a small CNN for demonstration.
        x = images
        # Resize and preprocess steps could be added here if needed.
        # For simplicity: use a Conv2D to get features shape (1, 64, 2048)
        conv = tf.keras.layers.Conv2D(2048, (3, 3), padding='same', activation='relu')
        features_map = conv(x)  # shape: (1, H, W, 2048)
        # Flatten spatial dims to 64 (for 8x8 spatial assumption)
        features_map_flat = tf.reshape(features_map, (batch_size, -1, 2048))  # (1, 64, 2048)

        # Encoder processes features to embedding_dim=256
        img_tensor_val = self.encoder(features_map_flat)  # (1, 64, 256)

        # Decoder initial hidden state
        hidden = self.decoder.reset_states(batch_size=batch_size)  # (1, 512)

        # Decoder input initialized to start token id
        dec_input = tf.expand_dims([self.start_token_id], 0)  # shape (1,1), dtype int32 by default
        dec_input = tf.cast(dec_input, tf.int64)  # Ensure input types conform to decoder signature

        result = tf.TensorArray(dtype=tf.int32, size=self.max_length)

        for i in tf.range(self.max_length):
            predictions, hidden, attention_weights = self.decoder(dec_input, img_tensor_val, hidden)
            # predictions shape (batch_size * 1, vocab_size) -> (1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=1)[0]  # greedy argmax for demonstration

            result = result.write(i, predicted_id)

            if tf.equal(predicted_id, self.end_token_id):
                # Stop early if end token predicted
                break

            # Next decoder input is predicted id
            dec_input = tf.expand_dims([predicted_id], 0)  # shape (1,1)
            dec_input = tf.cast(dec_input, tf.int64)

        output_seq = result.stack()  # (<= max_length,)
        return output_seq


def my_model_function():
    """
    Instantiate MyModel with example typical params.
    """
    # Typical embedding_dim=256, units=512, vocab_size=5000 (example)
    return MyModel(embedding_dim=256, units=512, vocab_size=5000, features_shape=2048)


def GetInput():
    """
    Produces a dummy image input tensor that matches the expected input to MyModel.
    Assumes images normalized to [0,1], shape (1, 299, 299, 3).
    """
    return tf.random.uniform(shape=(1, 299, 299, 3), dtype=tf.float32)

