# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Model hyperparameters as per the issue
        self.num_classes = 100
        self.input_shape_ = (32, 32, 3)

        self.image_size = 72  # resizing input images
        self.patch_size = 6
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 12*12 = 144
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_units = [self.projection_dim * 2, self.projection_dim]
        self.transformer_layers = 4
        self.mlp_head_units = [2048, 1024]

        # Data augmentation pipeline including normalization (to be adapted externally),
        # resizing, flipping, rotation and zoom.
        self.data_augmentation = tf.keras.Sequential([
            layers.Normalization(name="normalization"),  # requires adapt() externally
            layers.Resizing(self.image_size, self.image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ], name="data_augmentation")

        # Patch extraction layer
        self.patches_layer = Patches(self.patch_size)
        # Patch encoding layers
        self.patch_encoder = PatchEncoder(self.num_patches, self.projection_dim)

        # Transformer blocks: each consists of
        # LayerNorm -> MultiHeadAttention -> Residual Add
        # -> LayerNorm -> MLP block -> Residual Add
        self.transformer_layers_modules = []
        for _ in range(self.transformer_layers):
            mha = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1)
            ln1 = layers.LayerNormalization(epsilon=1e-6)
            ln2 = layers.LayerNormalization(epsilon=1e-6)
            mlp_block = MLP(self.transformer_units, dropout_rate=0.1)
            self.transformer_layers_modules.append((ln1, mha, ln2, mlp_block))

        self.final_layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.flatten = layers.Flatten()
        self.dropout_1 = layers.Dropout(0.5)
        self.mlp_head = MLP(self.mlp_head_units, dropout_rate=0.5)
        self.classifier = layers.Dense(self.num_classes)

    def call(self, inputs, training=False):
        # inputs assumed shape: (batch_size, 32, 32, 3) with dtype float32 (e.g. from GetInput)
        # Apply data augmentation pipeline
        x = self.data_augmentation(inputs, training=training)

        # Create patches and encode
        patches = self.patches_layer(x)
        encoded_patches = self.patch_encoder(patches)

        # Transformer blocks
        for ln1, mha, ln2, mlp_block in self.transformer_layers_modules:
            x1 = ln1(encoded_patches)
            attention_output = mha(x1, x1, training=training)
            x2 = attention_output + encoded_patches
            x3 = ln2(x2)
            x3 = mlp_block(x3, training=training)
            encoded_patches = x3 + x2

        # Final layers
        representation = self.final_layernorm(encoded_patches)
        representation = self.flatten(representation)
        representation = self.dropout_1(representation, training=training)
        features = self.mlp_head(representation, training=training)
        logits = self.classifier(features)
        return logits


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
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


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded


class MLP(layers.Layer):
    def __init__(self, hidden_units, dropout_rate):
        super(MLP, self).__init__()
        self.hidden_layers = []
        for units in hidden_units:
            self.hidden_layers.append(layers.Dense(units, activation=tf.nn.gelu))
            self.hidden_layers.append(layers.Dropout(dropout_rate))

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x


def my_model_function():
    # Create and return an instance of the ViT-like model wrapped inside MyModel
    return MyModel()


def GetInput():
    # Return a random float32 tensor input matching the input shape (batch, 32, 32, 3)
    # The model expects pixel values roughly in input range for normalization layer, so 0-255 scale consistent.
    batch_size = 8  # small batch size for example
    input_tensor = tf.random.uniform(
        shape=(batch_size, 32, 32, 3), minval=0, maxval=255, dtype=tf.float32
    )
    return input_tensor

