# tf.random.uniform((1, 256, 256, 3), dtype=tf.float32) ← inferred input shape and dtype from issue context

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original model is a modified ResNet101 with 3 outputs:
        #  - (1, 5) tensor (output_1)
        #  - (1, 24, 3) tensor (output_2)
        #  - (1, 24, 3) tensor (output_3)
        #
        # Since the original SavedModel lacks signatures and detailed layers,
        # and attempts to load & convert failed,
        # we reconstruct a plausible model approximating the output shapes,
        # keeping a ResNet101 backbone from keras.applications and
        # dummy heads producing outputs with the right shapes.
        #
        # This fused model helps convert to TFLite using a keras model as suggested.

        # Base backbone: ResNet101 without top, input shape matching input
        self.backbone = tf.keras.applications.ResNet101(
            include_top=False,
            weights=None,
            input_shape=(256, 256, 3),
            pooling='avg'  # global avg pool to flatten features
        )
        # Output 1 head: Dense to shape (5,)
        self.head1 = tf.keras.layers.Dense(5, activation=None, name='head1_dense')
        
        # Output 2 & 3 heads:
        # We simulate a sequence output of shape (24, 3).
        # Use Dense layers following a reshape of features.
        # For simplicity, replicate feature vector 24 times with a Dense layer to 3 dims.
        self.head2_dense = tf.keras.layers.Dense(3, activation=None, name='head2_dense')
        self.head3_dense = tf.keras.layers.Dense(3, activation=None, name='head3_dense')

        self.repeat_count = 24

    def call(self, inputs, training=False):
        # backbone features shape: (batch, feature_dim)
        features = self.backbone(inputs, training=training)  # shape (batch, feat_dim)
        
        # Output 1 
        out1 = self.head1(features)  # (batch, 5)

        # Repeat features for heads 2 and 3 to produce sequence outputs
        repeated_features = tf.repeat(tf.expand_dims(features, axis=1), repeats=self.repeat_count, axis=1)
        # repeated_features shape: (batch, 24, feat_dim)

        out2 = self.head2_dense(repeated_features)  # (batch, 24, 3)
        out3 = self.head3_dense(repeated_features)  # (batch, 24, 3)

        return (out1, out2, out3)

def my_model_function():
    # Return constructed instance of MyModel
    # No pretrained weights used here because original weights are not accessible.
    return MyModel()

def GetInput():
    # Return a random input tensor with shape (1, 256, 256, 3) matching model input specification.
    return tf.random.uniform((1, 256, 256, 3), dtype=tf.float32)

