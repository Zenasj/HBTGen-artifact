# tf.random.uniform((B, 3, 1024, 1024, 3), dtype=tf.float32) ← Input shape inferred from TimeDistributed input_shape=(3, 1024, 1024, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, classes_n=10):
        super().__init__()
        # This model fuses the original EfficientNetB0 backbone and the "cut" model to reflect the issue context.
        # Since pretrained weights or efn.EfficientNetB0 are not directly available here,
        # we construct representative placeholders for the purpose of reproducibility.

        # Assumption:
        # - "Full" EfficientNetB0 model (not including top) as a TimeDistributed submodel
        # - "Cut" EfficientNetB0 up to block6d_add as another TimeDistributed submodel
        # Both followed by pooling, dropout, dense, activation

        # For demonstration, we create two stub submodels with compatible shapes.
        # In practice, you'd replace these with efn.EfficientNetB0 instances.

        # Placeholder for full EfficientNetB0 backbone (simulated by a few Conv2D layers)
        # Output shape per time step ~ (32, 32, 320) approx from actual EfficientNetB0 after top=False
        self.full_model_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(320, 3, padding='same', activation='relu'),
        ])

        # Placeholder for "cut" model output at block6d_add: outputs shape roughly (64,64,672)
        self.cut_model_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(672, 3, padding='same', activation='relu'),
        ])

        # TimeDistributed wrappers for sequence input (3 time steps)
        self.full_td = tf.keras.layers.TimeDistributed(self.full_model_conv)
        self.cut_td = tf.keras.layers.TimeDistributed(self.cut_model_conv)

        # GlobalAveragePooling3D to pool across time and spatial dims
        self.global_pool = tf.keras.layers.GlobalAveragePooling3D()
        self.dropout = tf.keras.layers.Dropout(0.2)
        # Output dense layer - classes_n - 1 as per original code
        # To keep compatible, adding a small number of classes_n default 10
        self.dense = tf.keras.layers.Dense(classes_n - 1)
        self.activation = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')

    def call(self, inputs, training=False):
        """
        Forward pass:
          - Apply full EfficientNet model and cut model (TimeDistributed)
          - Apply global avg pooling, dropout, dense, activation on both
          - Compare outputs (e.g., check if close within tolerance)
          - Output the boolean tensor indicating comparison result for each sample
        """

        # inputs shape: (batch, 3, 1024, 1024, 3)

        # Full model output: shape (batch, 3, H_full, W_full, C_full)
        full_out = self.full_td(inputs)
        # Cut model output: shape (batch, 3, H_cut, W_cut, C_cut)
        cut_out = self.cut_td(inputs)

        # Pooling requires shape compatible with 3D pooling across time and spatial dims
        # The two outputs may differ in spatial dims, so we pool separately

        # We pad / crop to largest common shape or just pool independently and compare vectors

        # Apply GlobalAveragePooling3D per output (batch, channels)
        full_pooled = self.global_pool(full_out)
        cut_pooled = self.global_pool(cut_out)

        # Dense and activation layers for each branch outputs
        full_logits = self.activation(self.dense(self.dropout(full_pooled, training=training)))
        cut_logits = self.activation(self.dense(self.dropout(cut_pooled, training=training)))

        # Compare the output logits for similarity within tolerance
        tolerance = 1e-4
        comparison = tf.abs(full_logits - cut_logits) < tolerance
        # Return boolean tensor indicating if all classes agree for each batch element
        return tf.reduce_all(comparison, axis=-1)

def my_model_function():
    # Return an instance of MyModel with a reasonable number of classes, e.g., 10
    return MyModel(classes_n=10)

def GetInput():
    # Return a random float32 tensor matching expected input:
    # (batch_size, 3, 1024, 1024, 3)
    # Use batch_size = 2 for demonstration to avoid huge memory usage.
    batch_size = 2
    input_shape = (batch_size, 3, 1024, 1024, 3)
    return tf.random.uniform(input_shape, dtype=tf.float32)

