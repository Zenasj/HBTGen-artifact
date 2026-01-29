# tf.random.uniform((B, 20), dtype=tf.int32), tf.random.uniform((B, 4096), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras import layers, regularizers, backend as K, losses, Model, Input, optimizers

class MyModel(tf.keras.Model):
    def __init__(self, max_length=20, image_embed_size=4096, latent_dim=64,
                 reg_lambda=0.05, fnd_lambda=0.3, embedding_matrix=None):
        super().__init__()
        self.max_length = max_length
        self.image_embed_size = image_embed_size
        self.latent_dim = latent_dim
        self.reg_lambda = reg_lambda
        self.fnd_lambda = fnd_lambda

        self.vocab_size = embedding_matrix.shape[0]
        self.embedding_matrix = embedding_matrix

        # Encoder text embedding (non-trainable)
        self.txt_embed = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=32,
            input_length=self.max_length,
            weights=[self.embedding_matrix],
            trainable=False,
            name='txt_embed'
        )

        # Encoder text layers
        self.lstm_txt_1 = layers.Bidirectional(
            layers.LSTM(32, return_sequences=True, activation='tanh',
                        kernel_regularizer=regularizers.l2(self.reg_lambda)),
            merge_mode='concat', name='lstm_txt_1')
        self.lstm_txt_2 = layers.Bidirectional(
            layers.LSTM(32, return_sequences=False, activation='tanh',
                        kernel_regularizer=regularizers.l2(self.reg_lambda)),
            merge_mode='concat', name='lstm_txt_2')
        self.fc_txt = layers.Dense(32, activation='tanh',
                                   kernel_regularizer=regularizers.l2(self.reg_lambda),
                                   name='dense_txt')

        # Encoder image layers
        self.fc_img_1 = layers.Dense(1024, activation='tanh',
                                     kernel_regularizer=regularizers.l2(self.reg_lambda),
                                     name='fc_img_1')
        self.fc_img_2 = layers.Dense(32, activation='tanh',
                                     kernel_regularizer=regularizers.l2(self.reg_lambda),
                                     name='fc_img_2')

        # Shared layer after concatenation
        self.shared = layers.Dense(64, activation='tanh',
                                   kernel_regularizer=regularizers.l2(self.reg_lambda),
                                   name='shared')

        # Latent variables layers
        self.z_mean_layer = layers.Dense(self.latent_dim, activation='linear', name='z_mean')
        self.z_log_var_layer = layers.Dense(self.latent_dim, activation='linear', name='z_log_var')

        # Decoder text layers
        self.dec_fc_txt = layers.Dense(32, activation='tanh',
                                       kernel_regularizer=regularizers.l2(self.reg_lambda),
                                       name='dec_fc_txt')
        self.repeat_vec = layers.RepeatVector(self.max_length)
        self.dec_lstm_txt_1 = layers.LSTM(32, return_sequences=True, activation='tanh',
                                          kernel_regularizer=regularizers.l2(self.reg_lambda),
                                          name='dec_lstm_txt_1')
        self.dec_lstm_txt_2 = layers.LSTM(32, return_sequences=True, activation='tanh',
                                          kernel_regularizer=regularizers.l2(self.reg_lambda),
                                          name='dec_lstm_txt_2')
        self.decoded_txt_layer = layers.TimeDistributed(
            layers.Dense(self.vocab_size, activation='softmax'), name='decoded_txt')

        # Decoder image layers
        self.dec_fc_img_1 = layers.Dense(32, activation='tanh',
                                        kernel_regularizer=regularizers.l2(self.reg_lambda),
                                        name='dec_fc_img_1')
        self.dec_fc_img_2 = layers.Dense(1024, activation='tanh',
                                        kernel_regularizer=regularizers.l2(self.reg_lambda),
                                        name='dec_fc_img_2')
        self.decoded_img_layer = layers.Dense(4096, activation='sigmoid', name='decoded_img')

        # FND (fake news detector) layers
        self.fnd_dense_1 = layers.Dense(64, activation='tanh',
                                        kernel_regularizer=regularizers.l2(self.fnd_lambda))
        self.fnd_dense_2 = layers.Dense(32, activation='tanh',
                                        kernel_regularizer=regularizers.l2(self.fnd_lambda))
        self.fnd_output_layer = layers.Dense(1, activation='sigmoid', name='fnd_output')

    def sampling(self, z_mean, z_log_var):
        # Sampling z ~ N(z_mean, exp(z_log_var))
        batch = tf.shape(z_mean)[0]
        dim = self.latent_dim
        epsilon = tf.random.normal(shape=(batch, dim), mean=0., stddev=0.01)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs, training=False):
        # inputs: tuple of (text, image_embed)
        input_txt, input_img = inputs  # Expect shapes: (B, max_length), (B, 4096)

        # Encoder forward pass
        x_txt = self.txt_embed(input_txt)  # (B, max_length, 32)
        x_txt = self.lstm_txt_1(x_txt)     # (B, max_length, 64)
        x_txt = self.lstm_txt_2(x_txt)     # (B, 64)
        x_txt = self.fc_txt(x_txt)          # (B, 32)

        x_img = self.fc_img_1(input_img)   # (B, 1024)
        x_img = self.fc_img_2(x_img)       # (B, 32)

        x = tf.concat([x_txt, x_img], axis=-1)  # (B, 64)
        x = self.shared(x)  # (B, 64)

        z_mean = self.z_mean_layer(x)      # (B, latent_dim)
        z_log_var = self.z_log_var_layer(x) # (B, latent_dim)
        z = self.sampling(z_mean, z_log_var)  # (B, latent_dim)

        # Decoder forward pass (text)
        d_txt = self.dec_fc_txt(z)                # (B, 32)
        d_txt = self.repeat_vec(d_txt)            # (B, max_length, 32)
        d_txt = self.dec_lstm_txt_1(d_txt)        # (B, max_length, 32)
        d_txt = self.dec_lstm_txt_2(d_txt)        # (B, max_length, 32)
        decoded_txt = self.decoded_txt_layer(d_txt)  # (B, max_length, vocab_size)

        # Decoder forward pass (image)
        d_img = self.dec_fc_img_1(z)               # (B, 32)
        d_img = self.dec_fc_img_2(d_img)           # (B, 1024)
        decoded_img = self.decoded_img_layer(d_img)  # (B, 4096)

        # FND forward pass
        h = self.fnd_dense_1(z)
        h = self.fnd_dense_2(h)
        fnd_output = self.fnd_output_layer(h)     # (B, 1)

        # For loss calculation externally we need z_mean and z_log_var,
        # but Keras Model outputs expected to be tensors,
        # so include them in outputs for easy access (caller can slice if needed).
        # Here, we return only autoencoder outputs (3 heads): decoded_txt, decoded_img, fnd_output.

        return decoded_txt, decoded_img, fnd_output, z_mean, z_log_var

    def compute_loss(self, x_txt, x_img, y_txt, y_img, y_fnd):
        # Custom loss combining reconstruction loss and latent loss

        decoded_txt, decoded_img, fnd_output, z_mean, z_log_var = self((x_txt, x_img), training=True)

        # Reconstruction categorical crossentropy for text (sparse)
        txt_recon_loss = losses.sparse_categorical_crossentropy(y_true=y_txt, y_pred=decoded_txt)
        txt_recon_loss = tf.reduce_mean(txt_recon_loss)

        # Reconstruction mse loss for image
        img_recon_loss = tf.reduce_mean(tf.square(y_img - decoded_img), axis=-1)
        img_recon_loss = tf.reduce_mean(img_recon_loss)

        # KL Divergence latent loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )

        # Combined VAE losses for text and image decoders
        vae_loss_txt = txt_recon_loss + kl_loss
        vae_loss_img = img_recon_loss + kl_loss

        # Binary cross-entropy loss for fnd_output
        fnd_loss = losses.binary_crossentropy(y_true=y_fnd, y_pred=fnd_output)
        fnd_loss = tf.reduce_mean(fnd_loss)

        # Total loss: sum or weighted sum as per original code (weights managed externally)
        # Return all components for monitoring or weighted sum usage
        return vae_loss_txt, vae_loss_img, fnd_loss, decoded_txt, decoded_img, fnd_output

def my_model_function():
    # For demo purposes, initialize embedding_matrix to random uniform matrix
    # Assume vocab_size = 10000 to be compatible:
    import numpy as np
    vocab_size = 10000
    embedding_matrix = np.random.uniform(-0.05, 0.05, (vocab_size, 32)).astype('float32')

    return MyModel(max_length=20, image_embed_size=4096, latent_dim=64,
                   reg_lambda=0.05, fnd_lambda=0.3, embedding_matrix=embedding_matrix)


def GetInput():
    # Return a tuple compatible with MyModel inputs:
    # text input shape: (batch_size, max_length) with integer token indices [0,vocab_size-1]
    # image input shape: (batch_size, 4096), float32 values in [0,1] (normalized embeddings)

    batch_size = 8
    max_length = 20
    vocab_size = 10000
    text_input = tf.random.uniform(shape=(batch_size, max_length),
                                   minval=0,
                                   maxval=vocab_size,
                                   dtype=tf.int32)
    img_input = tf.random.uniform(shape=(batch_size, 4096),
                                  minval=0.0,
                                  maxval=1.0,
                                  dtype=tf.float32)
    return text_input, img_input

