# tf.random.uniform((None, 128, 128, 3), dtype=tf.float32) ← inferred input shape for MobileNet input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, dense_layers_units=[256], num_classes=10, initial_weights=True):
        super().__init__()
        # We use MobileNet without top, average pooling output, input shape 128x128x3
        # weights set to 'imagenet' only initially (simulate one-time init as in original code)
        self.initial_weights = initial_weights

        # Use MobileNet base model with or without weights.
        weights_flag = 'imagenet' if self.initial_weights else None
        self.mobilenet = tf.keras.applications.MobileNet(
            input_shape=(128, 128, 3),
            include_top=False,
            pooling='avg',
            weights=weights_flag
        )

        # After initialization, weights_flag set to None for later calls to prevent reload
        self.initial_weights = False

        # Fully connected dense layers as per params['dense_layers'] from original code
        self.dense_layers = []
        for units in dense_layers_units:
            self.dense_layers.append(tf.keras.layers.Dense(units=units, activation='relu'))

        # Final logits layer for 10 classes with ReLU activation (as in original code, though unusual)
        self.logits_layer = tf.keras.layers.Dense(units=num_classes, activation='relu')

    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: [B, 128, 128, 3], float32 in [0,1] range (assumed)
        returns: logits tensor [B, 10] (before softmax)
        """

        # Pass inputs through MobileNet base
        base_features = self.mobilenet(inputs, training=training)  # [B, embedding_dim]

        x = base_features
        # Pass through dense layers
        for dense in self.dense_layers:
            x = dense(x)

        logits = self.logits_layer(x)  # [B, 10]

        return logits


def my_model_function():
    # Return an instance of MyModel, simulating the preset params['dense_layers']=[256]
    # and initial_weights=True once on creation.
    # This emulates the once-load-weights logic by always loading weights on new model instantiation.
    return MyModel(dense_layers_units=[256], num_classes=10, initial_weights=True)


def GetInput():
    # Return a random input tensor compatible with MyModel input:
    # - shape [batch_size, 128, 128, 3], float32 values normalized to [0,1]
    # Batch size will be chosen arbitrarily as 8 for testing
    batch_size = 8
    input_tensor = tf.random.uniform(
        shape=(batch_size, 128, 128, 3),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32
    )
    return input_tensor

