# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assuming ResNet50 input shape is (None, 224, 224, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Set the mixed precision policy for the model; this would typically be set externally,
        # but here we fix to use bfloat16 for demonstration as per TPU mixed precision usage.
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.compat.v2.keras.mixed_precision.experimental.set_policy(policy)

        # Load base resnet50 model without weights for demonstration; no top included by default here.
        self.base_model = tf.keras.applications.ResNet50(weights=None)

        # Input layer must match base_model input shape
        # Create a new functional model that wraps base_model but ensures output is cast to float32
        # to avoid bfloat16 issues during loss/metrics calculations.
        inp = tf.keras.layers.Input(batch_shape=self.base_model.input_shape)
        out = self.base_model(inp)

        # Usually ResNet50 outputs logits (no softmax by default),
        # so cast logits output to float32 before final output.
        out = tf.keras.layers.Activation('linear', dtype='float32')(out)

        # Build the functional model for forward calls
        self.model = tf.keras.Model(inputs=inp, outputs=out)

    def call(self, inputs, training=False):
        # Forward through the wrapped functional model
        return self.model(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel with mixed_bfloat16 policy and float32 output
    return MyModel()

def GetInput():
    # ResNet50 default input shape is (batch_size, 224, 224, 3), dtype tf.float32 input
    # Use random uniform input (typical image input range [0,255] scaled to [0,1] is common)
    # We'll create float32 input, policy will cast internally as needed
    batch_size = 2  # arbitrary small batch size
    input_shape = (batch_size, 224, 224, 3)
    return tf.random.uniform(input_shape, dtype=tf.float32)

