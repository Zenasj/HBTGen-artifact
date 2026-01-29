# tf.random.uniform((batch_size, 2), dtype=tf.int32) ← input is pair of [user_id, item_id] indices, each a scalar int32

import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers

class MyModel(tf.keras.Model):
    def __init__(self,
                 num_users=1_000_000,
                 num_items=100_000,
                 mf_dim=10,
                 layers_units=[10],
                 reg_layers=[0.0],
                 reg_mf=0.0,
                 alpha=0.5):
        super().__init__()
        assert len(layers_units) == len(reg_layers), "Regularizers must match layers count"

        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.alpha = alpha
        self.layers_units = layers_units
        self.reg_layers = reg_layers

        # Embeddings for matrix factorization (MF) part
        self.mf_embedding_user = layers.Embedding(
            input_dim=num_users,
            output_dim=mf_dim,
            embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
            embeddings_regularizer=regularizers.l2(reg_mf),
            input_length=1,
            name='mf_embedding_user'
        )
        self.mf_embedding_item = layers.Embedding(
            input_dim=num_items,
            output_dim=mf_dim,
            embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
            embeddings_regularizer=regularizers.l2(reg_mf),
            input_length=1,
            name='mf_embedding_item'
        )

        # Embeddings for MLP part, output dim is half of first MLP layer units
        self.mlp_embedding_user = layers.Embedding(
            input_dim=num_users,
            output_dim=layers_units[0] // 2,
            embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
            embeddings_regularizer=regularizers.l2(reg_layers[0]),
            input_length=1,
            name='mlp_embedding_user'
        )
        self.mlp_embedding_item = layers.Embedding(
            input_dim=num_items,
            output_dim=layers_units[0] // 2,
            embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
            embeddings_regularizer=regularizers.l2(reg_layers[0]),
            input_length=1,
            name='mlp_embedding_item'
        )

        # Dense layers for MLP
        self.mlp_layers = []
        for idx in range(1, len(layers_units)):
            self.mlp_layers.append(
                layers.Dense(
                    units=layers_units[idx],
                    activation='relu',
                    kernel_regularizer=regularizers.l2(reg_layers[idx]),
                    bias_regularizer=regularizers.l2(reg_layers[idx]),
                    name=f"layer{idx}"
                )
            )

        # Final prediction layer (sigmoid for binary output)
        self.prediction_layer = layers.Dense(
            units=1,
            activation='sigmoid',
            kernel_initializer='lecun_uniform',
            bias_initializer='lecun_uniform',
            name='prediction'
        )

        self.flatten = layers.Flatten()
        self.multiply = layers.Multiply()
        self.concat = layers.Concatenate(axis=-1)

    def call(self, inputs, training=None):
        # inputs is tuple/list of (user_input, item_input)
        user_input, item_input = inputs  # both expected shape (batch_size, 1), dtype int32

        # MF part embeddings and elementwise multiply
        mf_user_latent = self.flatten(self.mf_embedding_user(user_input))  # (batch_size, mf_dim)
        mf_item_latent = self.flatten(self.mf_embedding_item(item_input))  # (batch_size, mf_dim)
        mf_vector = self.multiply([mf_user_latent, mf_item_latent])  # (batch_size, mf_dim)

        # MLP part embeddings and concatenation
        mlp_user_latent = self.flatten(self.mlp_embedding_user(user_input))  # (batch_size, layers[0]//2)
        mlp_item_latent = self.flatten(self.mlp_embedding_item(item_input))  # (batch_size, layers[0]//2)
        mlp_vector = self.concat([mlp_user_latent, mlp_item_latent])  # (batch_size, layers[0])

        # Forward through MLP dense layers
        for layer in self.mlp_layers:
            mlp_vector = layer(mlp_vector)

        # Scale MF and MLP parts by alpha and (1-alpha)
        mf_vector_scaled = mf_vector * self.alpha
        mlp_vector_scaled = mlp_vector * (1 - self.alpha)

        # Concatenate scaled parts
        predict_vector = self.concat([mf_vector_scaled, mlp_vector_scaled])

        # Final sigmoid prediction for interaction probability
        prediction = self.prediction_layer(predict_vector)  # (batch_size, 1)

        return prediction

def my_model_function():
    # Typical parameters from the original issue: very large user/item vocab with factorization and small MLP
    num_users = 1_000_000
    num_items = 100_000
    mf_dim = 10
    layers = [10]  # MLP with only input layer size 10
    reg_layers = [0.0]
    reg_mf = 0.0
    alpha = 0.5

    return MyModel(
        num_users=num_users,
        num_items=num_items,
        mf_dim=mf_dim,
        layers_units=layers,
        reg_layers=reg_layers,
        reg_mf=reg_mf,
        alpha=alpha
    )

def GetInput(batch_size=256):
    # Generate random batch of (user_id, item_id) pairs for the input
    num_users = 1_000_000
    num_items = 100_000

    # Random int32 tensor shape (batch_size, 1) each for user and item idx
    user_input = tf.random.uniform(
        shape=(batch_size, 1), minval=0, maxval=num_users, dtype=tf.int32
    )
    item_input = tf.random.uniform(
        shape=(batch_size, 1), minval=0, maxval=num_items, dtype=tf.int32
    )
    return (user_input, item_input)

