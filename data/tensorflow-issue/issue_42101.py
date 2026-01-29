# tf.random.uniform((None, 400), dtype=tf.float32) ← inferred input shape from the autoencoder example in the issue text

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Shapes and params inferred from the AE and HR model descriptions in the issue
        
        # Encoder submodel
        in_shape = (400,)
        latent_shape = (16,)
        self.encoder_in = layers.InputLayer(input_shape=in_shape, name="encoder_in")
        self.encoder_dense1 = layers.Dense(256, activation='relu', name="encoder_dense1")
        self.encoder_dense_out = layers.Dense(latent_shape[0], activation='relu', name="encoder_out")
        
        # Decoder submodel
        self.decoder_in = layers.InputLayer(input_shape=latent_shape, name="decoder_in")
        self.decoder_dense1 = layers.Dense(256, activation='relu', name="decoder_dense1")
        self.decoder_dense_out = layers.Dense(in_shape[0], name="decoder_out")
        
        # RNN model (HR model)
        # Using Lambda for expand dims as in example
        self.expand_dims = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name='expand_dims')
        self.rnn = layers.GRU(64, return_sequences=True, return_state=True, name="GRU_layer")
        self.conv1d = layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='same', activation=None, name='Conv1D')
        self.flatten = layers.Flatten()
        
        # Constants (based on example)
        self.HR_win_len = 200
        
    def call(self, inputs):
        # inputs shape: (batch_size, 400)
        
        # Encoder forward
        x = self.encoder_in(inputs)          # InputLayer, maintains shape
        x = self.encoder_dense1(x)
        latent = self.encoder_dense_out(x)  # latent representation shape (batch, 16)
        
        # Decoder forward
        y = self.decoder_in(latent)
        y = self.decoder_dense1(y)
        sig_hat = self.decoder_dense_out(y)  # Reconstructed signal shape (batch, 400)
        
        # HR model forward
        x_exp = self.expand_dims(sig_hat)  # shape (batch, 400, 1)
        
        # Slice x_exp for warm-up and main RNN run
        warmup_input = x_exp[:, :self.HR_win_len, :]  # (batch, 200, 1)
        main_input = x_exp[:, self.HR_win_len:, :]    # (batch, 200, 1)
        
        _, final_state = self.rnn(warmup_input)
        rnn_out, _ = self.rnn(main_input, initial_state=final_state)
        
        conv_out = self.conv1d(rnn_out)  # shape (batch, seq_len=200, filters=1)
        hr_hat = self.flatten(conv_out)  # flatten to (batch, 200)
        
        return hr_hat

def my_model_function():
    # Return an instance of the fused model
    return MyModel()

def GetInput():
    # Return a random tensor input matching (batch_size, 400), float32
    # Using batch size 2 as example; can be any batch size
    return tf.random.uniform((2, 400), dtype=tf.float32)

