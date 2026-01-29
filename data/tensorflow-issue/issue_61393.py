# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← typical CIFAR-10 input shape and type inferred from issue

import tensorflow as tf
from tensorflow.keras import layers

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'patch_size': self.patch_size})
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
        })
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(32, 32, 3), patch_size=4, projection_dim=64, num_patches=64, num_classes=10):
        super().__init__()
        self.input_shape_ = input_shape
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.num_classes = num_classes

        # Build the patch extractor and encoder layers
        self.patches = Patches(patch_size=self.patch_size)
        self.patch_encoder = PatchEncoder(num_patches=self.num_patches, projection_dim=self.projection_dim)

        # Simple Transformer encoder block placeholders:
        self.transformer_layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.transformer_mha = layers.MultiHeadAttention(num_heads=4, key_dim=self.projection_dim)
        self.transformer_dropout1 = layers.Dropout(0.1)
        self.transformer_dense1 = layers.Dense(2 * self.projection_dim, activation='relu')
        self.transformer_dense2 = layers.Dense(self.projection_dim)
        self.transformer_dropout2 = layers.Dropout(0.1)

        # Classifier head
        self.layernorm_head = layers.LayerNormalization(epsilon=1e-6)
        self.dense_head = layers.Dense(self.num_classes)

    def call(self, inputs, training=False):
        # inputs: (B, H, W, C)
        patches = self.patches(inputs)  # (B, num_patches, patch_dims)
        encoded_patches = self.patch_encoder(patches)  # (B, num_patches, projection_dim)

        # Transformer Encoder Block
        x1 = self.transformer_layernorm(encoded_patches)
        attention_output = self.transformer_mha(x1, x1)
        attention_output = self.transformer_dropout1(attention_output, training=training)
        x2 = attention_output + encoded_patches

        x3 = self.transformer_layernorm(x2)
        x3 = self.transformer_dense1(x3)
        x3 = self.transformer_dense2(x3)
        x3 = self.transformer_dropout2(x3, training=training)
        transformer_output = x3 + x2  # Residual connection

        # Classification head: global average pooling over patches
        representation = tf.reduce_mean(transformer_output, axis=1)
        representation = self.layernorm_head(representation)
        logits = self.dense_head(representation)
        return logits

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'input_shape_': self.input_shape_,
            'patch_size': self.patch_size,
            'projection_dim': self.projection_dim,
            'num_patches': self.num_patches,
            'num_classes': self.num_classes
        })
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def my_model_function():
    # For CIFAR-10 default image size is 32x32x3
    # Patch size chosen to be 4 -> 8x8 patches, but num_patches is set to 64 (assumed)
    return MyModel(input_shape=(32, 32, 3), patch_size=4, projection_dim=64, num_patches=64, num_classes=10)

def GetInput():
    # Return a random batch of images shaped (batch_size, 32, 32, 3)
    # Batch size example: 2
    batch_size = 2
    return tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32)

