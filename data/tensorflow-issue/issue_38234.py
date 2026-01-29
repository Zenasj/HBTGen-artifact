# tf.random.uniform((B, T, 13), dtype=tf.float32)  # B=batch size, T=timesteps, 13=feature dim

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, TimeDistributed, Dense

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Using two bidirectional LSTM layers with 128 units each and return_sequences=True
        self.b_lstm1 = Bidirectional(LSTM(128, return_sequences=True, implementation=2), input_shape=(None, 13))
        self.b_lstm2 = Bidirectional(LSTM(128, return_sequences=True, implementation=2))
        # TimeDistributed Dense layer to produce logits per timestep
        # The output dimension depends on the vocabulary size + 2 (for blank and possible additional tokens)
        # Here we set a placeholder vocabulary size, 
        # and user should replace `vocab_size` with actual vocab size used in `inv_mapping`
        self.vocab_size = 57  # inferred from label shape (59) minus 2, assumption on mapping
        self.tmd = TimeDistributed(Dense(self.vocab_size + 2))

    def call(self, x):
        x = self.b_lstm1(x)
        x = self.b_lstm2(x)
        x = self.tmd(x)
        return x

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, logits_time_major=False, reduction=tf.keras.losses.Reduction.SUM, name='ctc'):
        super().__init__(reduction=reduction, name=name)
        self.logits_time_major = logits_time_major

    def call(self, y_true, y_pred):
        # Cast labels to int32 as required by tf.nn.ctc_loss
        y_true = tf.cast(y_true, tf.int32)

        # Batch size
        batch_size = tf.shape(y_pred)[0]
        # Time steps (length of logits time dimension)
        time_steps = tf.shape(y_pred)[1]

        # Label length: assume labels are padded and take full label length for all batch elements
        label_length = tf.fill([batch_size], tf.shape(y_true)[1])
        # Logit length: assume logits length equals time_steps for all batch elements
        logit_length = tf.fill([batch_size], time_steps)

        # Compute CTC loss
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=-1
        )
        return tf.reduce_mean(loss)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random float32 tensor shaped (batch_size, time_steps, features)
    # Based on the issue: input_tensor was (2241, 1186, 13)
    # For demonstration and resource feasibility, use smaller batch/time sizes here:
    batch_size = 8  # smaller batch size for input generation
    time_steps = 1186
    feature_dim = 13
    return tf.random.uniform((batch_size, time_steps, feature_dim), dtype=tf.float32)

