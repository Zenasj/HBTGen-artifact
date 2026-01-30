import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
tflite_quantized_model = converter.convert()

class MySeq2SeqModel(tf.keras.models.Model):
    def __init__(self, vocab_size: int, input_len: int, output_len: int,
                 batch_size,
                 rnn_units: int = 64, dense_units: int = 64, embedding_dim: int = 256, **kwargs):
        super(MySeq2SeqModel, self).__init__(**kwargs)

        # Base Attributes
        self.vocab_size = vocab_size
        self.input_len = input_len
        self.output_len = output_len
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        # Beam search attributes
        self.beam_width = 3

        # Encoder
        self.encoder_embedding = layers.Embedding(vocab_size, embedding_dim, input_length=input_len)
        self.encoder_rnn = layers.LSTM(rnn_units, return_sequences=True, return_state=True)

        # Decoder
        self.decoder_embedding = layers.Embedding(vocab_size, embedding_dim, input_length=output_len)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)

        # Attention
        self.attention_mechanism = tfa.seq2seq.LuongAttention(dense_units)
        self.rnn_cell = self.build_rnn_cell(batch_size=batch_size)

        # Output
        self.dense_layer = tf.keras.layers.Dense(vocab_size)

        self.inference_decoder = BeamSearchDecoder(cell=self.rnn_cell,
                                                   beam_width=self.beam_width,
                                                   output_layer=self.dense_layer,
                                                   # As tf.nn.embedding_lookup is not supported by tflite
                                                   embedding_fn=lambda ids: tf.gather(tf.identity(
                                                       self.decoder_embedding.variables[0]), ids),
                                                   coverage_penalty_weight=0.0, dynamic=False, parallel_iterations=1,
                                                   maximum_iterations=output_len
                                                   )

    def call(self, inputs, training=None, mask=None):
        # Encoder
        encoder = self.encoder_embedding(inputs[0])
        encoder_outputs, state_h, state_c = self.encoder_rnn(encoder)
        decoder_emb = self.decoder_embedding(inputs[1])

        tiled_a = tfa.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_width)
        tiled_a_tx = tfa.seq2seq.tile_batch(state_h, multiplier=self.beam_width)
        tiled_c_tx = tfa.seq2seq.tile_batch(state_c, multiplier=self.beam_width)
        start_tokens = tf.fill([1], START_ID)

        self.attention_mechanism.setup_memory(tiled_a)

        final_output, final_state, _ = self.inference_decoder(embedding=None,
                                                              start_tokens=start_tokens,
                                                              end_token=EOS_ID,
                                                              initial_state=self.build_decoder_initial_state(
                                                                  size=1 * self.beam_width,
                                                                  encoder_state=[tiled_a_tx, tiled_c_tx],
                                                                  Dtype=tf.float32))

        return final_output.predicted_ids