import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)

    @tf.function(input_signature=(
        tf.TensorSpec([None, None], tf.int32, name='sequence'),
        (tf.TensorSpec([None, 64], tf.float32, name='states_1'), tf.TensorSpec([None, 64], tf.float32, name='states_2'))
    )) 
    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)

        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))

# Some more code for training the seq2seq model...

tf.saved_model.save(
    encoder,  # instance of Encoder
    './some/directory/',
    signatures=encoder.call
)