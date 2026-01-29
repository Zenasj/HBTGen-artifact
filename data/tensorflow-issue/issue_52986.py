# tf.random.uniform((B, 32, 32, 1), dtype=tf.float32) ← Input shape based on padded Fashion MNIST images with single channel

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Input layer: expects (32, 32, 1)
        self.input_layer = tf.keras.Input(shape=(32, 32, 1), name="image")

        # Convert single channel to 3 channels (ResNet50V2 requires 3 channels)
        self.preprocess_conv = tf.keras.layers.Conv2D(3, (1, 1), name="conv1x1_channel_expand")

        # Base backbone: ResNet50V2 without top, global average pooling
        self.backbone = tf.keras.applications.ResNet50V2(
            include_top=False,
            pooling="avg",
            weights=None,
            input_shape=(32, 32, 3),
        )

        # Dense layer for logits/classes (10 classes for Fashion MNIST)
        self.logits_layer = tf.keras.layers.Dense(10, name="dense_logits")

    def call(self, inputs, training=False):
        x = inputs
        # Convert 1 channel to 3 channels
        x = self.preprocess_conv(x)
        # Compute features via ResNet backbone
        features = self.backbone(x, training=training)
        # Dense logits layer (not necessarily used in original reproduction code's output, but typical)
        logits = self.logits_layer(features)
        return logits


def convert_to_sync_batch_norm(old_model: tf.keras.Model, input_layer: tf.keras.Input) -> tf.keras.Model:
    """
    Convert all BatchNormalization layers in a Keras model to SyncBatchNormalization.
    Rebuilds the model layer by layer, replacing BatchNorm layers.
    Args:
      old_model: tf.keras.Model instance to convert.
      input_layer: Input to the new model, tf.keras.Input.
    Returns:
      new_model: tf.keras.Model with SyncBatchNormalization layers.
    """
    old_layer_names = [layer.name for layer in old_model.layers]
    new_xs = [input_layer]
    for old_layer in old_model.layers[1:]:
        # Determine the input(s) for this layer in new model
        if isinstance(old_layer.input, list):
            input_x = [new_xs[old_layer_names.index(l.name.split("/")[0])] for l in old_layer.input]
        else:
            input_x = new_xs[old_layer_names.index(old_layer.input.name.split("/")[0])]

        # Replace BatchNormalization layers with SyncBatchNormalization
        if isinstance(old_layer, tf.keras.layers.BatchNormalization):
            # Recreate layer with SyncBatchNormalization config
            old_layer = tf.keras.layers.SyncBatchNormalization.from_config(old_layer.get_config())
        # Call layer with new input(s)
        x = old_layer(input_x)
        new_xs.append(x)

    # Build new Model
    new_model = tf.keras.Model(new_xs[0], new_xs[-1])

    # Transfer weights from old model layers to new model layers
    for old_layer, new_layer in zip(old_model.layers, new_model.layers):
        try:
            new_layer.set_weights(old_layer.get_weights())
        except ValueError:
            # Some layers may not have weights or mismatch - ignore these layers gracefully
            pass

    return new_model


def my_model_function(sync_batch_norm: bool = False):
    """
    Returns an instance of MyModel.
    If sync_batch_norm is True, converts ResNet backbone to use SyncBatchNormalization layers.
    """
    base_model = MyModel()

    if not sync_batch_norm:
        # Return as-is with standard BatchNormalization layers
        return base_model

    # Otherwise, convert backbone to use SyncBatchNormalization layers
    # We need input tensor for conversion
    input_shape = (32, 32, 1)
    input_layer = tf.keras.Input(shape=input_shape, name="image")
    # Expand channels from 1 to 3
    preprocess_conv = tf.keras.layers.Conv2D(3, (1, 1))(input_layer)
    # Create ResNet50V2 backbone with BatchNorm layers
    backbone_base = tf.keras.applications.ResNet50V2(
        include_top=False, pooling="avg", input_tensor=preprocess_conv, weights=None
    )
    backbone_model = tf.keras.Model(input_layer, backbone_base.output)

    # Convert BatchNormalization layers to SyncBatchNormalization
    new_backbone = convert_to_sync_batch_norm(backbone_model, input_layer=input_layer)

    class SyncBNModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.input_layer = input_layer
            self.backbone = new_backbone
            self.logits_layer = tf.keras.layers.Dense(10, name="dense_logits")

        def call(self, inputs, training=False):
            x = inputs
            features = self.backbone(x, training=training)
            logits = self.logits_layer(features)
            return logits

    return SyncBNModel()


def GetInput():
    """
    Returns a batch input tensor compatible with MyModel.
    The shape matches (batch_size, 32, 32, 1) with float32 values in [0,1].
    """
    batch_size = 4  # Assumed batch size for testing; can be adjusted
    # Generate random float input to mimic padded Fashion MNIST
    return tf.random.uniform(shape=(batch_size, 32, 32, 1), dtype=tf.float32)

