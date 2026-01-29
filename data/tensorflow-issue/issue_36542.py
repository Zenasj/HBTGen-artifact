# tf.random.uniform((batch_size, crop_size, crop_size, features), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assumptions based on partial info:
        # max_sequence_len and dictionary_size need reasonable defaults
        self.max_sequence_len = 20  # e.g., max length of sequence output
        self.dictionary_size = 1000  # vocab size representative
        self.num_rnn_units = 256

        # Define an LSTM layer as RNN layer component
        self.lstm = tf.keras.layers.LSTM(
            self.num_rnn_units,
            return_sequences=True,
            return_state=False,
            name="rnn_lstm"
        )
        # An output projection layer to map RNN output to dictionary probabilities
        self.dense = tf.keras.layers.Dense(self.dictionary_size, activation="softmax")

    def call(self, inputs, training=False):
        """
        inputs: tuple or dict containing:
            - selected_features: tensor of shape (batch, crop_size, crop_size, features)
            - true_words: tensor of shape (total_batches, max_sequence_len)
            - true_lengths: tensor of shape (total_batches,) sequence lengths
            - rnn_batch_size: integer batch size for RNN processing
        This emulates the RNNBatch class functionality on a batch of features.
        """
        selected_features = inputs['selected_features']
        true_words = inputs['true_words']
        true_lengths = inputs['true_lengths']
        rnn_batch_size = inputs.get('rnn_batch_size', 64)

        dtype = tf.float32
        batch_size = tf.shape(selected_features)[0]

        # Initialize states
        states_h = tf.zeros((batch_size, self.num_rnn_units), dtype=dtype)
        states_c = tf.zeros((batch_size, self.num_rnn_units), dtype=dtype)
        states = [states_h, states_c]

        # For simplicity, ignoring true_words and true_lengths in forward pass,
        # as in original the rnn_layer method uses them internally.
        # Instead, we simulate processing selected_features with LSTM + dense

        # Flatten spatial dims to a sequence for LSTM: (batch, time, features)
        # Assume selected_features shape is (batch, crop_size, crop_size, features)
        # We'll flatten crop_size * crop_size as "time" dimension
        shape_sf = tf.shape(selected_features)
        time_steps = shape_sf[1] * shape_sf[2]
        features = shape_sf[3]
        x = tf.reshape(selected_features, (batch_size, time_steps, features))

        # Run LSTM
        lstm_out = self.lstm(x, training=training)  # (batch, time_steps, num_rnn_units)

        # Project to dictionary size (probabilities)
        output_probs = self.dense(lstm_out)  # (batch, time_steps, dictionary_size)

        # For "outputs_ar" assume an alternate output with just argmax indices for some reason
        output_argmax = tf.argmax(output_probs, axis=-1, output_type=tf.int32)

        # The original class returns two outputs: out and out_ar.
        # Here we mimic both as separate outputs:
        return output_probs, output_argmax

    def infer_shape(self, input_shape):
        # Optional helper - not used here
        pass

def my_model_function():
    # Instantiate the MyModel
    return MyModel()

def GetInput():
    # Generate random input matching expected input to MyModel.call

    # Based on inferred input shape from comments and code:
    # 'selected_features' shape: (batch_size, crop_size, crop_size, features)
    batch_size = 128
    crop_size = 16  # chosen assuming small spatial crop
    features = 64   # feature channels

    selected_features = tf.random.uniform(
        (batch_size, crop_size, crop_size, features), dtype=tf.float32
    )

    max_sequence_len = 20
    total_batches = batch_size

    # true_words: integer indices, shape (total_batches, max_sequence_len)
    true_words = tf.random.uniform(
        (total_batches, max_sequence_len), minval=0, maxval=999, dtype=tf.int32
    )

    # true_lengths: actual sequence lengths, between 1 and max_sequence_len
    true_lengths = tf.random.uniform(
        (total_batches,), minval=1, maxval=max_sequence_len + 1, dtype=tf.int32
    )

    rnn_batch_size = 64  # typical batch size used in original code

    return {
        'selected_features': selected_features,
        'true_words': true_words,
        'true_lengths': true_lengths,
        'rnn_batch_size': rnn_batch_size
    }

