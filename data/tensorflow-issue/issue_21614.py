# tf.random.uniform((B, 50), dtype=tf.int32) ← inferred input shape based on Input(shape=(50,), dtype='int32')

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters inferred from the issue
        self.vocab_size = 50
        self.emb_act_size = 32
        self.rnn_act_size = 64
        self.out_size = 16

        # Layers as described in the original functional model
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.emb_act_size)
        # LSTM1 returns sequences plus state h and c
        self.lstm1 = tf.keras.layers.LSTM(
            self.rnn_act_size,
            return_sequences=True,
            return_state=True,
            implementation=2)
        # LSTM2 returns only sequences but uses initial_state from lstm1 state outputs
        self.lstm2 = tf.keras.layers.LSTM(
            self.rnn_act_size,
            return_sequences=True)
        # Final dense layer with softmax activation
        self.dense = tf.keras.layers.Dense(self.out_size, activation='softmax')

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, 50) integer sequences
        emb = self.embedding(inputs)  # Shape: (batch_size, 50, 32)
        lstm1_out, s_h, s_c = self.lstm1(emb, training=training)
        # Pass lstm1 outputs and initial_state explicitly to lstm2
        lstm2_out = self.lstm2(lstm1_out, initial_state=[s_h, s_c], training=training)
        y = self.dense(lstm2_out)
        return y

def my_model_function():
    # Instantiate and compile the model consistent with the original example
    model = MyModel()
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    return model

def GetInput():
    # Generate random integer tensor with shape (batch_size, 50)
    # using vocab_size 50 to simulate input sequences
    batch_size = 8  # Chosen arbitrarily - can be any positive integer
    seq_length = 50
    return tf.random.uniform(
        shape=(batch_size, seq_length),
        minval=0,
        maxval=50,
        dtype=tf.int32)

