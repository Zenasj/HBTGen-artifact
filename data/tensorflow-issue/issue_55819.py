# tf.random.uniform((BATCH_SIZE, 256, 256, 3), dtype=tf.float32) ← Input shape inferred from create_vit_classifier input

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Constants and hyperparameters (from the issue)
BUFFER_SIZE = 512  # Not used in model definition
BATCH_SIZE = 256
IMAGE_SIZE = 256
PATCH_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 50
LAYER_NORM_EPS = 1e-6
TRANSFORMER_LAYERS = 8
PROJECTION_DIM = 32
NUM_HEADS = 4
TRANSFORMER_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
MLP_HEAD_UNITS = [2048, 1024]

# Data augmentation normalization layer placeholder (needs .adapt() on training data externally)
data_augmentation = keras.Sequential(
    [layers.Normalization(name="normalization_layer")], name="data_augmentation"
)
# Note: You must call data_augmentation.layers[0].adapt(training_data) before training

# Diagonal attention mask as a constant for Locality Self-Attention
diag_attn_mask = 1 - tf.eye(NUM_PATCHES, dtype=tf.int8)
diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)  # Shape: (1, NUM_PATCHES, NUM_PATCHES)


class ShiftedPatchTokenization(layers.Layer):
    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        half_patch=PATCH_SIZE // 2,
        num_patches=NUM_PATCHES,
        projection_dim=PROJECTION_DIM,
        vanilla=False,
        **kwargs,
    ):
        super(ShiftedPatchTokenization, self).__init__(**kwargs)
        self.vanilla = vanilla  # Flag to switch to vanilla patch extractor
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = half_patch
        self.flatten_patches = layers.Reshape((num_patches, -1))
        self.projection = layers.Dense(units=projection_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "half_patch": self.half_patch,
                "vanilla": self.vanilla,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def crop_shift_pad(self, images, mode):
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:  # right-down
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad

    def call(self, images):
        if not self.vanilla:
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            tokens = self.projection(flat_patches)
        return tokens, patches


class PatchEncoder(layers.Layer):
    def __init__(
        self,
        num_patches=NUM_PATCHES,
        projection_dim=PROJECTION_DIM,
        **kwargs,
    ):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def get_config(self):
        config = super().get_config().copy()
        # position_embedding layer and positions tensor can't be serialized easily
        config.update({"num_patches": self.num_patches})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        return encoded_patches + encoded_positions


class MultiHeadAttentionLSA(layers.MultiHeadAttention):
    def __init__(self, tau=None, **kwargs):
        # tau is a trainable temperature scalar that scales attention logits
        super(MultiHeadAttentionLSA, self).__init__(**kwargs)
        # Initial tau value = sqrt of key_dim, trainable
        self.tau = tf.Variable(
            math.sqrt(float(self._key_dim)), trainable=True, name="tau"
        )

    def get_config(self):
        config = super().get_config().copy()
        # Include tau as scalar (detached numpy scalar)
        config.update({"tau": self.tau.numpy()})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        # Scaled by tau instead of sqrt(key_dim) fixed scalar
        scaled_query = query / self.tau
        attention_scores = tf.einsum(self._dot_product_equation, key, scaled_query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
        attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)
        return attention_output, attention_scores


def mlp(x, hidden_units, dropout_rate):
    # Multi-layer perceptron block with GELU activation and dropout
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class MyModel(tf.keras.Model):
    def __init__(self, vanilla=False):
        super(MyModel, self).__init__()
        self.vanilla = vanilla

        # Setup the data augmentation layer again here - expects adapted mean/std externally
        self.data_augmentation = keras.Sequential(
            [layers.Normalization(name="normalization_layer")], name="data_augmentation"
        )
        self.shifted_patch_tokenization = ShiftedPatchTokenization(vanilla=vanilla)
        self.patch_encoder = PatchEncoder()

        # Prepare transformer layers
        self.transformer_layers = []
        for _ in range(TRANSFORMER_LAYERS):
            layer_norm1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
            if not vanilla:
                mha = MultiHeadAttentionLSA(
                    num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
                )
            else:
                mha = layers.MultiHeadAttention(
                    num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
                )
            add1 = layers.Add()
            layer_norm2 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
            # MLP is applied during call, but define dropout as part of it
            self.transformer_layers.append(
                (layer_norm1, mha, add1, layer_norm2)
            )

        # MLP blocks after transformer stack:
        self.representation_norm = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
        self.flatten = layers.Flatten()
        self.dropout1 = layers.Dropout(0.5)
        self.mlp_head_units = MLP_HEAD_UNITS
        self.dropout2 = layers.Dropout(0.5)
        self.classifier = layers.Dense(4)  # Assumed 4 classes (from issue comment)

    def call(self, inputs):
        # Inputs shape: (batch_size, 256, 256, 3)
        x = self.data_augmentation(inputs)  # expects adapted Normalization layer

        tokens, _ = self.shifted_patch_tokenization(x)
        encoded_patches = self.patch_encoder(tokens)

        for (layer_norm1, mha, add1, layer_norm2) in self.transformer_layers:
            x1 = layer_norm1(encoded_patches)
            if not self.vanilla:
                attention_output = mha(x1, x1, attention_mask=diag_attn_mask)
            else:
                attention_output = mha(x1, x1)
            x2 = add1([attention_output, encoded_patches])
            x3 = layer_norm2(x2)
            x3 = mlp(x3, TRANSFORMER_UNITS, dropout_rate=0.1)
            encoded_patches = layers.Add()([x3, x2])

        representation = self.representation_norm(encoded_patches)
        representation = self.flatten(representation)
        representation = self.dropout1(representation)
        features = mlp(representation, self.mlp_head_units, dropout_rate=0.5)
        features = self.dropout2(features)
        logits = self.classifier(features)
        return logits


def my_model_function():
    # Returns an instance of MyModel
    # NOTE: You must call .adapt() on MyModel.data_augmentation.layers[0] externally with training data
    return MyModel()


def GetInput():
    # Return a random valid input tensor for MyModel: shape (BATCH_SIZE, 256, 256, 3), float32
    return tf.random.uniform(
        (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), minval=0.0, maxval=1.0, dtype=tf.float32
    )

