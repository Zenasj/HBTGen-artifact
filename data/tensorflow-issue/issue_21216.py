# tf.random.uniform((BATCH_SIZE, max_length_inp), dtype=tf.int32)
import tensorflow as tf
import numpy as np

tfe = tf.contrib.eager

def gru(units):
    # If you have a GPU, use CuDNNGRU, otherwise GRU with sigmoid recurrent activation.
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_activation='sigmoid',
                                  recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)  # (batch, seq_len) --> (batch, seq_len, embed_dim)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Attention layers
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # x shape: (batch, 1), hidden shape: (batch, dec_units), enc_output: (batch, seq_len, enc_units)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # (batch, 1, dec_units)
        score = tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))  # (batch, seq_len, dec_units)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)  # (batch, seq_len, 1)

        # Compute context vector as weighted sum of encoder output vectors
        context_vector = attention_weights * enc_output  # (batch, seq_len, enc_units)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, enc_units)

        x = self.embedding(x)  # (batch, 1, embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)  # (batch, 1, embedding_dim + enc_units)

        output, state = self.gru(x)  # output (batch, 1, dec_units), state (batch, dec_units)

        output = tf.reshape(output, (-1, output.shape[2]))  # (batch, dec_units)
        x = self.fc(output)  # (batch, vocab_size)

        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


class MyModel(tf.keras.Model):
    """
    Combined NMT model wrapper that holds encoder and decoder, optimizer and checkpoint,
    enabling forward passes and optionally restoration and saving of weights.
    """

    def __init__(self,
                 vocab_inp_size,
                 vocab_tar_size,
                 embedding_dim,
                 units,
                 batch_size,
                 learning_rate=0.01,
                 restore=False,
                 checkpoint_dir="./ckpt-py/",
                 restore_from="./ckpt-py/-1"):
        super(MyModel, self).__init__()
        self.batch_size = batch_size
        self.units = units
        self.encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
        # Use GradientDescent optimizer to avoid buggy Adam issues in TF 1.9 as discussed in issue
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.checkpoint_dir = checkpoint_dir
        self.restore_from = restore_from
        self.checkpoint = tfe.Checkpoint(optimizer=self.optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder,
                                         optimizer_step=tf.train.get_or_create_global_step())
        self.restore = restore
        
        # Try restoring weights immediately if restore=True
        if self.restore:
            rst = self.checkpoint.restore(self.restore_from)
            # assert_consumed might fail if model not called yet - ignore or catch error if needed
            try:
                rst.assert_consumed()
            except tf.errors.NotFoundError:
                # This can happen if some variables are missing; ignore for now as per issue discussion
                pass
            except AssertionError:
                # Same - skip for TF 1.9/1.10 checkpoint known issues
                pass

    def call(self, inputs, target=None, training=False):
        """
        Forward method to perform translation training step or inference.
        inputs: input batch tensor for encoder (batch_size, seq_len)
        target: target batch tensor for teacher forcing (batch_size, seq_len) used during training
        training: boolean, if True do teacher forcing and return loss, else do inference (greedy sampling)
        """
        if not training:
            # Inference mode: greedy decode using encoder + decoder
            # For simplicity, batch_size=inputs.shape[0]
            hidden = self.encoder.initialize_hidden_state()
            enc_output, enc_hidden = self.encoder(inputs, hidden)
            dec_hidden = enc_hidden
            batch_size = inputs.shape[0]
            dec_input = tf.expand_dims([self.decoder.embedding.input_dim - 1] * batch_size, 1)  # Usually <start> token idx

            # We do not have vocab/targ_lang here, so skip detailed decoding and just do 1 step to get logits
            # To align with usage, here return logits for the first decoder step - partial inference.
            # In practice you'd write a loop to generate tokens step-by-step.
            logits, dec_hidden, attn_weights = self.decoder(dec_input, dec_hidden, enc_output)
            return logits  # (batch_size, vocab_tar_size)

        else:
            # Training mode - do a full teacher-forcing pass and return average loss
            loss = 0
            hidden = self.encoder.initialize_hidden_state()
            enc_output, enc_hidden = self.encoder(inputs, hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims(target[:, 0], 1)  # first input to decoder is <start> token
            seq_len = target.shape[1]

            for t in range(1, seq_len):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += self.loss_function(target[:, t], predictions)
                dec_input = tf.expand_dims(target[:, t], 1)  # teacher forcing

            batch_loss = loss / tf.cast(seq_len, tf.float32)
            return batch_loss

    def loss_function(self, real, pred):
        # Calculate loss masked on padding (=0)
        mask = 1 - tf.cast(tf.equal(real, 0), dtype=tf.float32)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    def initialize_hidden_state(self):
        return self.encoder.initialize_hidden_state()

    def save_checkpoint(self):
        save_path = self.checkpoint.save(self.checkpoint_dir)
        return save_path

    def restore_checkpoint(self):
        rst = self.checkpoint.restore(self.restore_from)
        try:
            rst.assert_consumed()
        except (tf.errors.NotFoundError, AssertionError):
            # Known issues with checkpoint restoration in TF 1.9, ignore these errors
            pass

def GetInput():
    """
    Produce a random integer tensor simulating a batch input sequence to the model.
    Shape: (BATCH_SIZE, sequence_length) with integer token indices in input vocab range.
    Assumptions made:
    - vocab_inp_size, BATCH_SIZE, max_length_inp known or inferred
    """
    # Use example sizes from the data pre-processing in the issue:
    BATCH_SIZE = 64
    max_length_inp = 20  # Approximate max sequence length, inferred from code context
    vocab_inp_size = 1000  # Placeholder: large enough vocab size for testing
    
    input_tensor = tf.random.uniform(
        shape=(BATCH_SIZE, max_length_inp),
        minval=1,
        maxval=vocab_inp_size,
        dtype=tf.int32
    )
    return input_tensor

def my_model_function():
    # Same assumptions as above for vocab sizes and architecture params
    BATCH_SIZE = 64
    embedding_dim = 256
    units = 1024
    vocab_inp_size = 1000
    vocab_tar_size = 1000

    model = MyModel(
        vocab_inp_size=vocab_inp_size,
        vocab_tar_size=vocab_tar_size,
        embedding_dim=embedding_dim,
        units=units,
        batch_size=BATCH_SIZE,
        learning_rate=0.01,
        restore=False,
        checkpoint_dir="./ckpt-py/",
        restore_from="./ckpt-py/-1"
    )
    return model

