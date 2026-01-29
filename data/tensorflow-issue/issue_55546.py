# tf.random.uniform((B, 7, 25, 81), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Layer
from tensorflow.keras.initializers import TruncatedNormal
import numpy as np

class PositionalEmbedding(Layer):
    def __init__(self, units, dropout_rate, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.units = units
        self.projection = Dense(units, kernel_initializer=TruncatedNormal(stddev=0.02))
        self.dropout = Dropout(rate=dropout_rate)

    def build(self, input_shape):
        super(PositionalEmbedding, self).build(input_shape)
        # input_shape expected: (batch_size, seq_len=7, dim2=25, input_dim=81)
        # position shape: (1, seq_len, dim2, units)
        self.position = self.add_weight(
            name="position",
            shape=(1, input_shape[1], input_shape[2], self.units),
            initializer=TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs, training=None):
        # inputs shape: (batch, 7, 25, 81)
        x = self.projection(inputs)
        x += self.position
        return self.dropout(x, training=training)

class Transformer(tf.keras.Model):
    def __init__(self, embed_dim, dropout_rate, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.pos_embs = PositionalEmbedding(embed_dim, dropout_rate)

    def call(self, inputs, training=None):
        # inputs is expected to be a tuple/list: (inputs_tensor, targets_inputs_tensor)
        # but from the issue context, only inputs tensor is passed in model call.
        # To keep compatibility, Support call with inputs tensor only (positional embedding on inputs)
        if isinstance(inputs, (tuple, list)):
            inputs_tensor = inputs[0]
        else:
            inputs_tensor = inputs
        return self.pos_embs(inputs_tensor, training=training)

    def train_step(self, data):
        inputs, targets = data

        # targets_inputs are targets shifted by one timestep (teacher forcing)
        targets_inputs = targets[:, :-1]
        targets_real = targets[:, 1:]

        with tf.GradientTape() as tape:
            y_pred = self([inputs, targets_inputs], training=True)
            loss = self.compiled_loss(targets_real, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(targets_real, y_pred)
        return {m.name: m.result() for m in self.metrics}

def my_model_function():
    return Transformer(embed_dim=256, dropout_rate=0.1)

def GetInput():
    # Generate a random tensor matching the expected input shape
    # Batch size 4, sequence length 7, 25, 81 features
    # Use float32 dtype matching model expectation
    return tf.random.uniform((4, 7, 25, 81), dtype=tf.float32)

