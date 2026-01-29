# tf.random.uniform((BATCH_SIZE, max_length_output), dtype=tf.float32) ← input shape for decoder call input
import tensorflow as tf
import tensorflow_addons as tfa

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm_layer = tf.keras.layers.LSTM(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, h, c = self.lstm_layer(x, initial_state=hidden)
        return output, h, c

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, max_length_input, max_length_output, attention_type='luong'):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type
        self.max_length_output = max_length_output

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Attention mechanism with memory to be set later
        self.attention_mechanism = self.build_attention_mechanism(
            self.dec_units,
            None,
            self.batch_sz * [max_length_input],
            self.attention_type
        )

        self.rnn_cell = self.build_rnn_cell(batch_sz)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(
            self.decoder_rnn_cell,
            self.attention_mechanism,
            attention_layer_size=self.dec_units
        )
        return rnn_cell

    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
        if attention_type == 'bahdanau':
            return tfa.seq2seq.BahdanauAttention(
                units=dec_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length
            )
        else:
            return tfa.seq2seq.LuongAttention(
                units=dec_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length
            )

    def build_initial_state(self, batch_sz, encoder_state, dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=dtype)
        # clone with encoder hidden state as cell state (h,c)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):
        # inputs shape: (batch_size, max_length_output)
        x = self.embedding(inputs)  # (batch_size, max_length_output, embedding_dim)
        outputs, _, _ = self.decoder(
            x,
            initial_state=initial_state,
            sequence_length=self.batch_sz * [self.max_length_output - 1]
        )
        return outputs

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, unit1, unit2, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(unit1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(unit2, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

class SourceModel(tf.keras.Model):
    def __init__(self):
        super(SourceModel, self).__init__()
        self.layer = tf.keras.layers.Dense(128, activation='relu')
        self.custom_layer = CustomLayer(64, 10)

    def call(self, inputs):
        x = self.layer(inputs)
        x = self.custom_layer(x)
        return x

class TargetModel(tf.keras.Model):
    def __init__(self):
        super(TargetModel, self).__init__()
        self.layer = tf.keras.layers.Dense(128, activation='relu')
        # Different output size (5) for target model causing shape mismatch on reload
        self.custom_layer = CustomLayer(64, 5)

    def call(self, inputs):
        x = self.layer(inputs)
        x = self.custom_layer(x)
        return x

class MyModel(tf.keras.Model):
    """
    Fused model encapsulating both SourceModel and TargetModel from the example,
    with functionality to compare their forward pass outputs elementwise.

    This illustrates reusing weights where possible (e.g., 'layer'),
    but custom_layer differs (output units differ), so only compatible weights load partially.

    Forward pass returns a boolean tensor indicating where outputs match within tolerance.
    """
    def __init__(self):
        super(MyModel, self).__init__()
        self.source_model = SourceModel()
        self.target_model = TargetModel()

    def call(self, inputs):
        # Forward pass through both models
        source_out = self.source_model(inputs)
        target_out = self.target_model(inputs)

        # Due to different last layer output sizes (10 vs 5), slice source_out to match target_out shape
        # This simulates partial weight compatibility scenario discussed in the issue
        source_out_slice = source_out[..., :target_out.shape[-1]]

        # Compare outputs numerically - here we check if difference is small (tolerance 1e-5)
        diff = tf.abs(source_out_slice - target_out)
        comparison = diff < 1e-5
        # Return boolean tensor showing per-element closeness
        return comparison

def my_model_function():
    # Instantiate MyModel and return
    model = MyModel()
    # It would be reasonable to run a build call to create weights if needed here,
    # but inputs with shape known from GetInput() call suffice.
    return model

def GetInput():
    # The SourceModel and TargetModel expect input shape (?, feature_dim)
    # From example SourceModel.layer Dense(128) means input features can be arbitrary.
    # Let's assume feature_dim=20 for example.
    feature_dim = 20
    batch_size = 64  # Based on BATCH_SIZE used in examples

    # Return random float32 tensor of shape (batch_size, feature_dim)
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)

