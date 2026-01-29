# tf.random.uniform((B, Tx, xFeatures), dtype=tf.float32) for encoder_inputs
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

class MyModel(tf.keras.Model):
    def __init__(self, n_a, n_s, Tx, Ty, xFeatures, yFeatures):
        super().__init__()
        self.n_a = n_a
        self.n_s = n_s
        self.Tx = Tx
        self.Ty = Ty
        self.xFeatures = xFeatures
        self.yFeatures = yFeatures

        # Encoder LSTMs
        self.encoder_lstm1 = LSTM(n_a, return_sequences=True, return_state=True)
        self.encoder_lstm2 = LSTM(n_a, return_state=True)

        # Repeat vector for decoder input
        self.repeat_vector = RepeatVector(Ty)

        # Decoder LSTMs
        self.decoder_lstm1 = LSTM(n_s, return_sequences=True, return_state=True)
        self.decoder_lstm2 = LSTM(n_s, return_sequences=True, return_state=True)

        # Final TimeDistributed Dense layer with relu activation, glorot init, and l2 reg
        self.dense_time_dist = TimeDistributed(
            Dense(yFeatures, activation='relu',
                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                  kernel_regularizer=regularizers.l2(0.01))
        )

    def call(self, inputs, training=False):
        """
        inputs: a list/tuple of 9 tensors, as per original functional model:
            [encoder_inputs (B, Tx, xFeatures),
             s1 (B, n_a), c1 (B, n_a),
             s2 (B, n_a), c2 (B, n_a),
             s3 (B, n_s), c3 (B, n_s),
             s4 (B, n_s), c4 (B, n_s)]
        returns: list of multiple outputs:
            decoder_outputs, hiddenState1, cellState1,
            hiddenState2, cellState2,
            hiddenState3, cellState3,
            hiddenState4, cellState4
        """
        encoder_inputs = inputs[0]
        s1, c1 = inputs[1], inputs[2]
        s2, c2 = inputs[3], inputs[4]
        s3, c3 = inputs[5], inputs[6]
        s4, c4 = inputs[7], inputs[8]

        encoder_lstm1_out, hiddenState1, cellState1 = self.encoder_lstm1(
            encoder_inputs, initial_state=[s1, c1], training=training)
        encoder_lstm2_out, hiddenState2, cellState2 = self.encoder_lstm2(
            encoder_lstm1_out, initial_state=[s2, c2], training=training)

        repeat_vector_out = self.repeat_vector(encoder_lstm2_out)

        decoder_lstm1_out, hiddenState3, cellState3 = self.decoder_lstm1(
            repeat_vector_out, initial_state=[s3, c3], training=training)
        decoder_lstm2_out, hiddenState4, cellState4 = self.decoder_lstm2(
            decoder_lstm1_out, initial_state=[s4, c4], training=training)

        decoder_outputs = self.dense_time_dist(decoder_lstm2_out)

        return [
            decoder_outputs,
            hiddenState1, cellState1,
            hiddenState2, cellState2,
            hiddenState3, cellState3,
            hiddenState4, cellState4
        ]

def my_model_function():
    # Hardcoded parameters as per the original example
    n_a = 64
    n_s = 64
    n_past = 40   # Tx
    n_future = 20 # Ty
    xFeatures = 11
    yFeatures = 6
    return MyModel(n_a, n_s, n_past, n_future, xFeatures, yFeatures)

def GetInput():
    """
    Returns inputs that match the MyModel expectation:
    List of 9 inputs:
     - encoder_inputs: shape (B, Tx, xFeatures) float32
     - s1, c1, s2, c2: shape (B, n_a) float32 zeros (encoder LSTM states)
     - s3, c3, s4, c4: shape (B, n_s) float32 zeros (decoder LSTM states)
    """
    n_a = 64
    n_s = 64
    Tx = 40
    xFeatures = 11
    batch_size = 32

    encoder_inputs = tf.random.uniform((batch_size, Tx, xFeatures), dtype=tf.float32)
    zero_s1 = tf.zeros((batch_size, n_a), dtype=tf.float32)
    zero_c1 = tf.zeros((batch_size, n_a), dtype=tf.float32)
    zero_s2 = tf.zeros((batch_size, n_a), dtype=tf.float32)
    zero_c2 = tf.zeros((batch_size, n_a), dtype=tf.float32)
    zero_s3 = tf.zeros((batch_size, n_s), dtype=tf.float32)
    zero_c3 = tf.zeros((batch_size, n_s), dtype=tf.float32)
    zero_s4 = tf.zeros((batch_size, n_s), dtype=tf.float32)
    zero_c4 = tf.zeros((batch_size, n_s), dtype=tf.float32)

    return [encoder_inputs, zero_s1, zero_c1, zero_s2, zero_c2, zero_s3, zero_c3, zero_s4, zero_c4]

