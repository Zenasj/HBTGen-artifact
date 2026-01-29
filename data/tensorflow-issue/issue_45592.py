# tf.random.uniform((500, 10, 50), dtype=tf.float32) ← inferred input shape from X in the issue

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, GRU, Dense, BatchNormalization
from tensorflow.keras import regularizers

class MyModel(tf.keras.Model):
    def __init__(
        self,
        nodes=15,
        activation="selu",
        kernel_initializer='random_uniform',
        regularizer_l1=0.05,
        regularizer_l2=0.0,
        recurrent_dropout=0.0,
        dropout=0.0,
        dense_units=(),
        batchnorm=True,
        batchnorm_trainable=True,
        batchnorm_training=False,
        use_bias=False,
    ):
        super().__init__()
        self.dropout_layer = Dropout(dropout) if dropout else None
        self.gru = GRU(
            nodes,
            activation=activation,
            recurrent_dropout=recurrent_dropout,
            return_sequences=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizers.l1_l2(regularizer_l1, regularizer_l2),
            use_bias=use_bias,
        )
        self.dense_layers = []
        for n in dense_units:
            self.dense_layers.append(
                Dense(
                    n,
                    activation=activation,
                    kernel_regularizer=regularizers.l1_l2(0.01, 0.01),
                    name=f'extra_dense{n}'
                )
            )
        self.batchnorm = BatchNormalization(trainable=batchnorm_trainable) if batchnorm else None
        self.batchnorm_training = batchnorm_training
        self.prediction_dense = Dense(
            1,
            activation='linear',
            use_bias=use_bias,
            kernel_initializer='random_uniform',
            name='prediction'
        )

    def call(self, inputs, training=False):
        x = inputs
        if self.dropout_layer:
            x = self.dropout_layer(x, training=training)
        x = self.gru(x, training=training)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        if self.batchnorm:
            # batchnorm training flag as from argument or fixed
            x = self.batchnorm(x, training=self.batchnorm_training if not training else training)
        x = self.prediction_dense(x)
        return x


def my_model_function():
    # Build an instance with default parameters matching those used in the issue
    return MyModel(
        nodes=15,
        activation="selu",
        kernel_initializer='random_uniform',
        regularizer_l1=0.05,
        regularizer_l2=0.0,
        recurrent_dropout=0.0,
        dropout=0.0,
        dense_units=(),
        batchnorm=True,
        batchnorm_trainable=True,
        batchnorm_training=False,
        use_bias=False,
    )


def GetInput():
    # Returning a random tensor matching the input shape used in the issue and dtype float32
    # Shape based on (n_samples, n_timesteps, n_features) = (500, 10, 50)
    return tf.random.uniform(shape=(500, 10, 50), dtype=tf.float32)

